#!/usr/bin/env python
# coding=utf-8

import pdb
import numpy as np
import shutil
import os
import glob
import time
from copy import deepcopy
import argparse

import matplotlib.pyplot as plt

from vtk.util.numpy_support import vtk_to_numpy as v2n

from svfsi import svFSI, sv_names
from post import main_arg

from utilities import QRfiltering_mod


class FSG(svFSI):
    """
    FSG-specific stuff
    """

    def __init__(self, f_params=None):
        # svFSI simulations
        svFSI.__init__(self, f_params)

    def run_post(self):
        # todo: read in automatically
        self.err = np.load(
            "study_lab_meeting/fsi_res_2022-11-30_18-21-39.375658/err.npy",
            allow_pickle=True,
        ).item()
        self.p["f_out"] = "."
        self.plot_convergence()

    def run(self):
        # run simulation
        try:
            self.main()
        except KeyboardInterrupt:
            print("interrupted")
            pass

        # archive results
        self.archive()

        # plot convergence
        self.plot_convergence()

        # post process
        main_arg([self.p["f_out"]])

    def main(self):
        # print reynolds number
        print("Re = " + str(int(self.p["re"])))

        # loop load steps
        i = 0
        for t in range(self.p["nmax"] + 1):
            print(
                "=" * 30
                + " t "
                + str(t)
                + " ==== fp "
                + "{:.2f}".format(self.p_vec[t])
                + " "
                + "=" * 30
            )

            # predict solution for next load step
            if t > 0:
                self.coup_predict(i, t)

            # loop sub-iterations
            for n in range(self.p["coup"]["nmax"]):
                # count total iterations (load + sub-iterations)
                i += 1

                # perform coupling step
                times = {}
                if self.p["coup"]["method"] in ["static", "aitken"]:
                    status = self.coup_step_relax(i, t, n, times)
                elif self.p["coup"]["method"] == "iqn_ils":
                    status = self.coup_step_iqn_ils(i, t, n, times)
                else:
                    raise ValueError(
                        "Unknown coupling method " + self.p["coup"]["method"]
                    )

                # check if simulation failed
                for name, s in self.curr.sol.items():
                    if s is None:
                        print(name + " simulation failed")
                        return

                # screen output
                out = "i " + str(i - 1) + " \tn " + str(n) + "\t"
                for name, e in self.err.items():
                    out += "{:.2e}".format(e[-1][-1]) + "\t"
                if self.p["coup"]["method"] in ["static", "aitken"]:
                    for name, e in self.p["coup"]["omega"].items():
                        out += "{:.2e}".format(e[-1][-1]) + "\t"
                for f in times.keys():
                    out += "{:.2e}".format(times[f]) + "\t"

                # check if coupling unconverged (screen and file output)
                if n == self.p["coup"]["nmax"] - 1:
                    out += "\n\tcoupling unconverged"
                    status = True
                print(out)

                # archive solution
                dst = os.path.join(self.p["f_sim"], "tube_" + str(i).zfill(3) + ".vtu")
                self.curr.archive("tube", dst)

                # check if coupling converged
                if status:
                    # save converged steps
                    i_conv = str(i).zfill(3)
                    t_conv = str(t).zfill(3)

                    srcs = os.path.join(self.p["f_sim"], "*_" + i_conv + ".*")
                    for src in glob.glob(srcs):
                        trg = os.path.basename(src).replace(i_conv, t_conv)
                        trg = os.path.join(self.p["f_conv"], trg)
                        shutil.copyfile(src, trg)

                    # archive
                    self.converged += [self.curr.copy()]

                    # terminate coupling
                    break

    def plot_convergence(self):
        if not self.err:
            print("no convergence data to plot (simulation may have failed before first coupling iteration)")
            return
        n_sol = len(self.err.keys())
        col_err = "k"
        col_omg = "r"

        n_iter = [0]
        fig, ax = plt.subplots(
            n_sol, 1, figsize=(20, 4), dpi=200, sharex="all", sharey="all"
        )
        for i, name in enumerate(self.err.keys()):
            # get_axis handle
            if n_sol == 1:
                axi = ax
            else:
                axi = ax[i]

            # get iteration counts
            if i == 0:
                n_iter += [len(res) for res in self.err[name]]
                n_iter = np.cumsum(n_iter)

            # second axis for omega
            if self.p["coup"]["method"] in ["static", "aitken"]:
                ax2 = axi.twinx()

            # collect results
            for j, res in enumerate(self.err[name]):
                # iteration numbers
                x = np.arange(n_iter[j], n_iter[j + 1])

                # plot error
                axi.plot(x, res, linestyle="-", color=col_err)

                # plot omega
                if self.p["coup"]["method"] in ["static", "aitken"]:
                    ax2.plot(x, self.p["coup"]["omega"][name][j], color=col_omg)

            # plot convergence criterion
            axi.plot([0, n_iter[-1]], self.p["coup"]["tol"] * np.ones(2), "k--")

            # axis settings
            axi.tick_params(axis="y", colors=col_err)
            axi.set_xticks(
                n_iter[1:] - 1,
                [
                    "$t_{" + str(i) + "}$, n=" + str(j)
                    for i, j in enumerate(np.diff(n_iter))
                ],
            )
            axi.set_xticks(np.arange(0, n_iter[-1]), minor=True)
            axi.set_xlim([0, n_iter[-1]])
            axi.set_ylabel("Residual " + sv_names[name], color=col_err)
            axi.set_yscale("log")
            axi.set_ylim([self.p["coup"]["tol"] * 0.1, 10])
            axi.grid(which="minor", alpha=0.2)
            axi.grid(which="major", alpha=0.9)
            if i == len(self.err.keys()) - 1:
                axi.set_xlabel("Number of iterations $n$ per time step $t$")

            if self.p["coup"]["method"] in ["static", "aitken"]:
                ax2.tick_params(axis="y", colors=col_omg)
                ax2.set_ylabel("Omega", color=col_omg)
                ax2.set_ylim([0.0, 1.0])
                ax2.set_yticks(np.linspace(0, 1, 6))

            axi.set_title("Total iterations: " + str(n_iter[-1]))

        # save to file
        fig.savefig(
            os.path.join(self.p["f_out"], "convergence.png"), bbox_inches="tight"
        )
        # plt.show()
        plt.close(fig)

    def archive(self):
        # save debug QR data if enabled
        if self.p["coup"].get("iqn_ils_debug", False):
            np.save(os.path.join(self.p["f_out"], "debug_qr.npy"), self.debug_qr)

        # save stored results
        self.p["error"] = self.err

        # save parameters
        self.save_params(self.p["name"] + ".json")

        # save input files
        for src in self.p["inp"].values():
            trg = os.path.join(self.p["f_arx"], os.path.basename(src))
            shutil.copyfile(os.path.join(self.p["paths"]["in_svfsi"], src), trg)

        # save python scripts
        sp = os.path.dirname(os.path.realpath(__file__))
        for src in ["fsg.py", "svfsi.py"]:
            trg = os.path.join(self.p["f_arx"], os.path.basename(src))
            shutil.copyfile(os.path.join(sp, src), trg)

        # save material model
        f_code = os.path.join(
            self.p["paths"]["exe"], os.path.split(self.p["exe"]["solid"])[0]
        )
        cpp_files = f_code + "/../../../Code/Source/svFSI/gr_*.*"
        for src in glob.glob(cpp_files):
            trg = os.path.join(self.p["f_arx"], os.path.basename(src))
            shutil.copyfile(src, trg)

    def coup_step_iqn_ils(self, i, t, n, times):
        # step 0: mesh movement (not in first first iteration)
        if self.p["fsi"] and i > 1:
            if self.step("mesh", i, t, n, times):
                return False
        else:
            times["mesh"] = 0.0

        # store previous solutions
        self.prev = self.curr.copy()

        solid_centered = self.p["coup"].get("solid_centered", False)

        if solid_centered:
            # solid-centered: R(tau) = F(S(tau)) - tau = 0

            # initialize wss from Poiseuille flow before very first solid step
            # (in fluid-centered mode fluid always runs first, so wss is never NaN
            # when solid runs; in solid-centered we must bootstrap it here)
            if i == 1:
                self.poiseuille(t)

            # step 1: solid update
            if self.step("solid", i, t, n, times):
                return False

            # step 2: fluid update
            if self.p["fsi"]:
                if self.step("fluid", i, t, n, times):
                    return False
            else:
                self.poiseuille(t)
        else:
            # fluid-centered (default): R(d) = S(F(d)) - d = 0
            # step 1: fluid update
            if self.p["fsi"]:
                if self.step("fluid", i, t, n, times):
                    return False
            else:
                self.poiseuille(t)

            # step 2: solid update
            if self.step("solid", i, t, n, times):
                return False

        # select fixed-point variable depending on formulation
        if solid_centered:
            # solid-centered: fixed-point variable is wss at fluid interface
            fp_kind_curr = ("fluid", "wss", "int")
            fp_kind_prev = ("fluid", "wss", "int")
            fp_key = "wss"
        else:
            # fluid-centered: fixed-point variable is solid displacement at interface
            fp_kind_curr = ("solid", "disp", "int")
            fp_kind_prev = ("solid", "disp", "int")
            fp_key = "disp"

        # log interface solution
        dtk = deepcopy(self.curr.get(fp_kind_curr)).flatten()
        dk = deepcopy(self.prev.get(fp_kind_prev)).flatten()

        # store increments
        # todo: save memory by only storing necessary information
        self.dk[fp_key] += [dtk]
        self.res += [dtk - dk]

        # append difference vectors after preloading (must not span different time levels)
        if t > 0 and n > 0 and len(self.dk[fp_key]) >= 2:
            self.mat_W += [self.dk[fp_key][-1] - self.dk[fp_key][-2]]
            self.mat_V += [self.res[-1] - self.res[-2]]

        # get error
        self.coup_err("solid", fp_key, i, t, n, scalar_res=solid_centered)

        # relax update
        self.coup_omega(fp_key, i, t, n)
        if not self.coup_converged(n):
            # no IQN-ILS update during preloading or first time step
            if ((t == 0) or (t == 1 and n < 5)):# or n == 0:
                if solid_centered:
                    self.coup_relax_wss(i, t, n)
                else:
                    self.coup_relax("solid", "disp", i, t, n)
            else:
                # reset history vectors at start of each new load step if requested
                if n == 0 and self.p["coup"].get("iqn_ils_reset", False):
                    self.mat_V = []
                    self.mat_W = []
                    # clear dk and res entirely so no cross-step difference vectors
                    # are built at n=1 (dk needs at least 2 entries from the same
                    # load step before mat_W/mat_V can be populated)
                    self.dk[fp_key] = []
                    self.res = []

                # fall back to relaxation if no history vectors available yet
                if not self.mat_V:
                    if solid_centered:
                        self.coup_relax_wss(i, t, n)
                    else:
                        self.coup_relax("solid", "disp", i, t, n)
                    return

                # maximum number of time steps used in IQN-ILS
                nq = self.p["coup"]["iqn_ils_q"]

                # trim to max number of considered vectors
                self.mat_V = self.mat_V[-nq:]
                self.mat_W = self.mat_W[-nq:]

                # remove linearly dependent vectors
                tmp_V = np.array(self.mat_V).T
                tmp_W = np.array(self.mat_W).T
                eps = self.p["coup"]["iqn_ils_eps"]

                # debug: save V and W before filtering
                if self.p["coup"].get("iqn_ils_debug", False):
                    self.debug_qr["V_before"] += [tmp_V.copy()]
                    self.debug_qr["W_before"] += [tmp_W.copy()]
                    self.debug_qr["ncols_before"] += [tmp_V.shape[1]]

                qq, rr, tmp_V, tmp_W = QRfiltering_mod(tmp_V, tmp_W, eps)

                # debug: save Q, R and V, W after filtering
                if self.p["coup"].get("iqn_ils_debug", False):
                    self.debug_qr["Q"] += [qq.copy()]
                    self.debug_qr["R"] += [rr.copy()]
                    self.debug_qr["V_after"] += [tmp_V.copy()]
                    self.debug_qr["W_after"] += [tmp_W.copy()]
                    self.debug_qr["ncols_after"] += [tmp_V.shape[1]]

                # solve for coefficients
                ss = np.dot(np.transpose(qq), -self.res[-1])
                cc = np.linalg.solve(rr, ss)

                # debug: save coefficients and load step/sub-iter indices
                if self.p["coup"].get("iqn_ils_debug", False):
                    self.debug_qr["cc"] += [cc.copy()]
                    self.debug_qr["t"] += [t]
                    self.debug_qr["n"] += [n]

                # safety check: if ||cc|| is too large the IQN update is ill-conditioned;
                # fall back to relaxation for this step to avoid divergence
                cc_max = self.p["coup"].get("iqn_ils_cc_max", np.inf)
                if np.linalg.norm(cc) > cc_max:
                    if solid_centered:
                        self.coup_relax_wss(i, t, n)
                    else:
                        self.coup_relax("solid", "disp", i, t, n)
                    return

                # update fixed-point variable
                vec_new = dtk + np.dot(tmp_W, cc)
                if solid_centered:
                    corr = np.dot(tmp_W, cc)
                    p95 = np.percentile(dtk[dtk > 0], 95) if np.any(dtk > 0) else np.nan
                    print(f"  [IQN-wss] t={t} n={n} ||cc||={np.linalg.norm(cc):.3e} "
                          f"dtk=[{dtk.min():.3e},{dtk.max():.3e}] "
                          f"corr=[{corr.min():.3e},{corr.max():.3e}] "
                          f"vec_new=[{vec_new.min():.3e},{vec_new.max():.3e}] "
                          f"n_neg={int(np.sum(vec_new<0))} n_spike={int(np.sum(vec_new>2*p95))}")
                    # wss is scalar — write directly to bypass add()'s vector norm call
                    map_v = self.curr.sim.map((("int", "fluid"), ("vol", "tube")))
                    self.curr.sol["wss"][map_v] = vec_new
                    map_src = self.curr.sim.map((("vol", "solid"), ("int", "fluid")))
                    map_trg = self.curr.sim.map((("vol", "solid"), ("vol", "tube")))
                    self.curr.sol["wss"][map_trg] = vec_new[map_src]
                else:
                    self.curr.add(("solid", "disp", "int"), vec_new.reshape((-1, 3)))

                # store matrices
                self.mat_V = tmp_V.T.tolist()
                self.mat_W = tmp_W.T.tolist()
        else:
            return True

    def coup_step_relax(self, i, t, n, times):
        # step 0: mesh movement (not in very first iteration)
        if self.p["fsi"] and i > 1:
            if self.step("mesh", i, t, n, times):
                return False

        # store previous solutions
        self.prev = self.curr.copy()

        solid_centered = self.p["coup"].get("solid_centered", False)

        if solid_centered:
            # solid-centered: R(tau) = F(S(tau)) - tau = 0
            # step 1: solid update
            if self.step("solid", i, t, n, times):
                return False

            # step 2: fluid update
            if self.p["fsi"]:
                if self.step("fluid", i, t, n, times):
                    return False
            else:
                self.poiseuille(t)
        else:
            # fluid-centered (default): R(d) = S(F(d)) - d = 0
            # step 1: fluid update
            if self.p["fsi"]:
                if self.step("fluid", i, t, n, times):
                    return False
            else:
                self.poiseuille(t)

            # step 2: solid update
            if self.step("solid", i, t, n, times):
                return False

        # log interface solution for aitken relaxation
        dtk = deepcopy(self.curr.get(("solid", "disp", "int"))).flatten()
        dk = deepcopy(self.prev.get(("solid", "disp", "int"))).flatten()
        self.dk["disp"] += [dtk]
        self.res += [dtk - dk]

        # calculate new relaxation factor
        self.coup_omega("disp", i, t, n)

        # get error
        self.coup_err("solid", "disp", i, t, n)

        # relax solid update
        if not self.coup_converged(n):
            self.coup_relax("solid", "disp", i, t, n)
        else:
            return True

    def coup_predict(self, i, t):
        # predict displacements
        kind = ("solid", "disp", "vol")

        if t == 0 or not self.p["predict_file"]:
            # extrapolate from previous time step(s)
            sol = self.predictor(kind, t)
        else:
            # predict from file
            sol = self.predictor_tube(kind, t)
        self.curr.add(kind, sol)

        # in solid-centered mode, predict WSS by extrapolating from converged history
        # (same strategy as displacement: carry forward or linearly extrapolate)
        # This ensures curr holds a consistent WSS at load step start rather than
        # stale end-of-previous-step sub-iteration WSS
        if self.p["coup"].get("solid_centered", False):
            wss_kind = ("fluid", "wss", "vol")
            n_sol = len(self.converged)
            if n_sol == 0:
                # no history yet: Poiseuille is the only option
                self.poiseuille(t)
            else:
                # reuse or extrapolate from converged WSS — same logic as displacement
                wss_pred = self.predictor(wss_kind, t)
                # write scalar wss directly (bypasses add()'s vector norm call)
                map_v = self.curr.sim.map((("vol", "fluid"), ("vol", "tube")))
                self.curr.sol["wss"][map_v] = wss_pred

    def predictor(self, kind, t):
        # fluid, solid, tube
        # disp, velo, wss, press
        # vol, int
        d, f, p = kind

        # number of old solutions
        n_sol = len(self.converged)

        if n_sol == 0:
            if f == "disp":
                # zero displacements
                return np.zeros(self.points[(p, d)].shape)
            elif f == "wss":
                # wss from poiseuille flow through reference configuration
                self.poiseuille(t)
                return self.curr.get(kind)
            else:
                raise ValueError("No predictor for field " + f)

        # previous solution
        vec_m0 = self.converged[-1].get(kind)
        if n_sol == 1:
            return vec_m0

        # linearly extrapolate from previous load increment
        vec_m1 = self.converged[-2].get(kind)
        # if n_sol == 2:
        return 2.0 * vec_m0 - vec_m1

        # quadratically extrapolate from previous two load increments
        # vec_m2 = self.converged[-3].get(kind)
        # return 3.0 * vec_m0 - 3.0 * vec_m1 + vec_m2

    def predictor_tube(self, kind, t):
        d, f, p = kind
        fname = "gr_partitioned/tube_" + str(t).zfill(3) + ".vtu"
        # fname = 'gr/gr_' + str(t + 1).zfill(3) + '.vtu'
        if not os.path.exists(fname):
            return None
        geo = read_geo(fname).GetOutput()
        if f == "disp":
            return v2n(geo.GetPointData().GetArray("Displacement"))[
                self.map(((p, d), ("vol", "tube")))
            ]
        elif f == "wss":
            if geo.GetPointData().HasArray("WSS"):
                return v2n(geo.GetPointData().GetArray("WSS"))[
                    self.map(((p, d), ("vol", "tube")))
                ]
            else:
                disp = v2n(geo.GetPointData().GetArray("Displacement"))[
                    self.map(((p, d), ("vol", "solid")))
                ]
                self.curr.add((d, "disp", p), disp)
                self.poiseuille(t)
                return self.curr.get(kind)

    def coup_relax(self, domain, name, i, t, n):
        # volume increment
        curr_v = deepcopy(self.curr.get((domain, name, "vol")))
        prev_v = deepcopy(self.prev.get((domain, name, "vol")))

        # relax update
        if i == 1:
            vec_relax = curr_v
        else:
            omega = self.p["coup"]["omega"][name][-1][-1]
            vec_relax = omega * curr_v + (1.0 - omega) * prev_v

        # update solution
        self.curr.add((domain, name, "vol"), vec_relax)

        # log interface solution for aitken relaxation
        dk = deepcopy(self.curr.get((domain, name, "int"))).flatten()
        self.dtk[name] += [dk]

    def coup_relax_wss(self, i, t, n):
        # relaxation for wss in solid-centered formulation
        # wss is stored as a scalar magnitude; write directly to avoid add()'s norm call
        curr_w = deepcopy(self.curr.get(("fluid", "wss", "int")))
        prev_w = deepcopy(self.prev.get(("fluid", "wss", "int")))

        if i == 1:
            wss_relax = curr_w
        else:
            omega = self.p["coup"]["omega"]["wss"][-1][-1]
            wss_relax = omega * curr_w + (1.0 - omega) * prev_w

        p95 = np.percentile(curr_w[curr_w > 0], 95) if np.any(curr_w > 0) else np.nan
        print(f"  [relax-wss] t={t} n={n} omega={omega if i>1 else 1.0:.3f} "
              f"curr=[{curr_w.min():.3e},{curr_w.max():.3e}] "
              f"prev=[{prev_w.min():.3e},{prev_w.max():.3e}] "
              f"relaxed=[{wss_relax.min():.3e},{wss_relax.max():.3e}] "
              f"n_neg={int(np.sum(wss_relax<0))} n_spike={int(np.sum(wss_relax>2*p95))}")
        # write relaxed scalar wss directly (bypasses add()'s norm call for vectors)
        map_v = self.curr.sim.map((("int", "fluid"), ("vol", "tube")))
        self.curr.sol["wss"][map_v] = wss_relax
        # propagate to solid volume (wss assumed constant radially)
        map_src = self.curr.sim.map((("vol", "solid"), ("int", "fluid")))
        map_trg = self.curr.sim.map((("vol", "solid"), ("vol", "tube")))
        self.curr.sol["wss"][map_trg] = wss_relax[map_src]

        # log interface wss for relaxation tracking
        dk = deepcopy(self.curr.get(("fluid", "wss", "int"))).flatten()
        self.dtk["wss"] += [dk]

    def coup_err(self, domain, name, i, t, n, scalar_res=False):
        if i == 1:
            # first step: no old solution
            err = 1.0
        elif scalar_res:
            # scalar fixed-point variable (e.g. wss in solid-centered formulation)
            err = np.mean(np.abs(self.res[-1]))
        else:
            # vector fixed-point variable (e.g. displacement): mean nodal L2-norm
            err = np.mean(np.linalg.norm(self.res[-1].reshape((-1, 3)), axis=1))

        # start a new sub-list for new load step
        if n == 0:
            self.err[name].append([])

        # append error norm
        self.err[name][-1].append(err)

    def coup_converged(self, n):
        # check if coupling converged
        check_tol = np.all(
            np.array([e[-1][-1] for e in self.err.values()]) < self.p["coup"]["tol"]
        )
        check_n = n >= self.p["coup"]["nmin"]
        return check_tol and check_n

    def coup_omega(self, name, i, t, n):
        # no relaxation necessary during prestressing (prestress does not depend on wss)
        if t == 0:
            omega = 1.0
        else:
            # static relaxation or first step of new load step
            omega = self.p["coup"]["omega0"]

            # dynamic relaxation
            if self.p["coup"]["method"] == "aitken" and n > 0:
                kuettler = True
                if kuettler:
                    rki = self.res[-1]
                    rkm = self.res[-2]
                    diff = rki - rkm
                    omega = (
                        -self.p["coup"]["omega"][name][-1][-1]
                        * np.dot(rkm, diff)
                        / np.dot(diff, diff)
                    )
                else:
                    # get old relaxed solutions
                    dp = self.dk[name][-1]
                    dk = self.dk[name][-2]

                    # get old unrelaxed solutions
                    dtk = self.dtk[name][-1]
                    dtm = self.dtk[name][-2]

                    # aitken update
                    diff = dtk - dp - dtm + dk
                    omega = np.dot(dtk - dtm, diff) / np.dot(diff, diff)

                # lower bound
                omega = np.max([omega, 0.1])

                # upper bound
                omega = np.min([omega, 1.0])

        # start a new sub-list for new load step
        if n == 0:
            self.p["coup"]["omega"][name].append([])

        # append
        self.p["coup"]["omega"][name][-1].append(omega)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an equilibrated Fluid-Solid-Growth interaction simulation (FSGe)"
    )
    parser.add_argument("sim", help="simulation parameters (.json)")
    parser.add_argument("-post", action="store_true", help="post-process only")
    args = parser.parse_args()

    fsg = FSG(args.sim)
    if args.post:
        fsg.run_post()
    else:
        fsg.run()
