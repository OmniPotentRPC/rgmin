// eOn client L-BFGS and FIRE (bench/eon_ref.cpp, TheochemUI/eOn 579738a)
// on the same Lennard-Jones and Morse systems as examples/large_n.rs.
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
constexpr double kMaxMove = 0.2;
constexpr double kFd = 0.01;
constexpr double kInv = 0.01;
constexpr double kEps = 1e-30;
struct Fg {
    virtual ~Fg() = default;
    virtual void eval(const std::vector<double> &x, double &e, std::vector<double> &g) = 0;
};
double dot(const std::vector<double> &a, const std::vector<double> &b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

double norm(const std::vector<double> &a) { return std::sqrt(dot(a, a)); }

double max_atom(const std::vector<double> &v) {
    double m = 0.0;
    const size_t n = v.size();
    size_t i = 0;
    for (; i + 3 <= n; i += 3) {
        const double r = std::sqrt(v[i] * v[i] + v[i + 1] * v[i + 1] + v[i + 2] * v[i + 2]);
        m = std::max(m, r);
    }
    if (i < n) {
        double r2 = 0.0;
        for (; i < n; ++i) {
            r2 += v[i] * v[i];
        }
        m = std::max(m, std::sqrt(r2));
    }
    return m;
}

std::vector<double> cap_atom(std::vector<double> v, double cap) {
    const double m = max_atom(v);
    if (m > cap && m > 0.0) {
        const double s = cap / m;
        for (double &x : v) {
            x *= s;
        }
    }
    return v;
}

std::vector<double> scale(const std::vector<double> &v, double s) {
    std::vector<double> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        o[i] = s * v[i];
    }
    return o;
}

std::vector<double> add(const std::vector<double> &a, const std::vector<double> &b) {
    std::vector<double> o(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        o[i] = a[i] + b[i];
    }
    return o;
}

std::vector<double> sub(const std::vector<double> &a, const std::vector<double> &b) {
    return add(a, scale(b, -1.0));
}

double stop_grad(const std::vector<double> &g, bool cluster) {
    if (!cluster) {
        double m = 0.0;
        for (double v : g) {
            m = std::max(m, std::abs(v));
        }
        return m;
    }
    return max_atom(g);
}

struct Counters {
    int evals = 0;
};

void call(Fg &fg, Counters &c, const std::vector<double> &x, double &e, std::vector<double> &g) {
    fg.eval(x, e, g);
    c.evals += 1;
}

struct Lbfgs {
    int memory = 20;
    std::deque<std::vector<double>> s, y;
    std::deque<double> rho;
    std::vector<double> r_prev, f_prev;
    int iteration = 0;

    void reset() {
        s.clear();
        y.clear();
        rho.clear();
    }

    std::vector<double> get_step(Fg &fg, Counters &c, const std::vector<double> &r,
                                 const std::vector<double> &force, double max_move) {
        double h0 = kInv;
        if (iteration > 0) {
            const std::vector<double> dr = sub(r, r_prev);
            const std::vector<double> yv = sub(f_prev, force);
            const double sy = dot(dr, yv);
            const double yy = dot(yv, yv);
            const double curv = (sy != 0.0 && std::isfinite(sy) && std::isfinite(yy)) ? yy / sy : -1.0;
            if (curv < 0.0) {
                reset();
                return cap_atom(scale(force, 1000.0), max_move);
            }
            if (sy > 0.0 && yy > 0.0) {
                h0 = sy / yy;
            }
        } else {
            const double fn = norm(force);
            if (fn > 0.0) {
                const std::vector<double> u = scale(force, 1.0 / fn);
                const std::vector<double> probe = add(r, scale(u, kFd));
                double ep = 0.0;
                std::vector<double> gp(r.size());
                call(fg, c, probe, ep, gp);
                std::vector<double> dg = gp;
                for (size_t i = 0; i < dg.size(); ++i) {
                    dg[i] += force[i];
                }
                const double curv = dot(dg, u) / kFd;
                if (!(curv > 0.0) || !std::isfinite(curv)) {
                    reset();
                    return cap_atom(scale(force, 1000.0), max_move);
                }
                h0 = 1.0 / curv;
            }
        }

        std::vector<double> q(force.size());
        for (size_t i = 0; i < force.size(); ++i) {
            q[i] = -force[i];
        }
        const int n = static_cast<int>(s.size());
        std::vector<double> alpha(static_cast<size_t>(n));
        for (int k = n - 1; k >= 0; --k) {
            alpha[static_cast<size_t>(k)] = rho[static_cast<size_t>(k)] * dot(s[static_cast<size_t>(k)], q);
            const double a = alpha[static_cast<size_t>(k)];
            for (size_t i = 0; i < q.size(); ++i) {
                q[i] -= a * y[static_cast<size_t>(k)][i];
            }
        }
        for (double &v : q) {
            v *= h0;
        }
        for (int k = 0; k < n; ++k) {
            const double b = rho[static_cast<size_t>(k)] * dot(y[static_cast<size_t>(k)], q);
            const double coef = alpha[static_cast<size_t>(k)] - b;
            for (size_t i = 0; i < q.size(); ++i) {
                q[i] += coef * s[static_cast<size_t>(k)][i];
            }
        }
        std::vector<double> d(q.size());
        for (size_t i = 0; i < q.size(); ++i) {
            d[i] = -q[i];
        }
        if (max_atom(d) >= max_move) {
            reset();
            return cap_atom(scale(force, h0), max_move);
        }
        const double dn = norm(d);
        const double fn = norm(force);
        if (dn > 0.0 && fn > 0.0 && dot(d, force) < 0.0) {
            reset();
            return cap_atom(scale(force, h0), max_move);
        }
        return cap_atom(d, max_move);
    }

    void update(const std::vector<double> &r1, const std::vector<double> &r0,
                const std::vector<double> &f1, const std::vector<double> &f0) {
        std::vector<double> s0 = sub(r1, r0);
        std::vector<double> y0 = sub(f0, f1);
        const double sy = dot(s0, y0);
        if (!std::isfinite(sy) || std::abs(sy) < kEps) {
            reset();
            return;
        }
        rho.push_back(1.0 / sy);
        s.push_back(std::move(s0));
        y.push_back(std::move(y0));
        while (static_cast<int>(s.size()) > memory) {
            s.pop_front();
            y.pop_front();
            rho.pop_front();
        }
    }

    void step(Fg &fg, Counters &c, std::vector<double> &x, double & /*e*/, std::vector<double> &g,
              double max_move) {
        std::vector<double> force(g.size());
        for (size_t i = 0; i < g.size(); ++i) {
            force[i] = -g[i];
        }
        if (iteration > 0) {
            update(x, r_prev, force, f_prev);
        }
        const std::vector<double> dr = get_step(fg, c, x, force, max_move);
        r_prev = x;
        f_prev = force;
        x = add(x, dr);
        double e = 0.0;
        call(fg, c, x, e, g);
        iteration += 1;
    }
};

struct Fire {
    double dt = 1.0;
    double dt_max = 2.5;
    double alpha = 0.1;
    double alpha0 = 0.1;
    int nmin = 5;
    int n = 0;
    std::vector<double> vel;
    bool stalled = false;

    void step(Fg &fg, Counters &c, std::vector<double> &x, std::vector<double> &g, double max_move) {
        if (vel.empty()) {
            vel.assign(x.size(), 0.0);
        }
        std::vector<double> force(g.size());
        for (size_t i = 0; i < g.size(); ++i) {
            force[i] = -g[i];
            vel[i] += force[i] * dt;
        }
        std::vector<double> dx = cap_atom(scale(vel, dt), max_move);
        x = add(x, dx);
        double e = 0.0;
        call(fg, c, x, e, g);
        for (size_t i = 0; i < g.size(); ++i) {
            force[i] = -g[i];
        }
        const double fn = norm(force);
        const double vn = norm(vel);
        const double p = dot(force, vel);
        if (fn > 0.0) {
            const double scale_v = alpha * vn / fn;
            for (size_t i = 0; i < vel.size(); ++i) {
                vel[i] = (1.0 - alpha) * vel[i] + scale_v * force[i];
            }
        }
        if (p >= 0.0) {
            n += 1;
            if (n > nmin) {
                dt = std::min(dt * 1.1, dt_max);
                alpha *= 0.99;
            }
        } else {
            dt *= 0.5;
            std::fill(vel.begin(), vel.end(), 0.0);
            alpha = alpha0;
            n = 0;
        }
        if (dt < 1e-6) {
            stalled = true;
        }
    }
};
struct System {
    std::string name;
    bool morse = false;
    int atoms = 0;
    std::vector<double> x0;
    double cell[3] = {0, 0, 0};
    double cutoff = 0;
};

struct Cells {
    std::vector<int> head, next;
};

double mic(double d, double boxl) {
    if (boxl > 0.0) {
        return d - boxl * std::round(d / boxl);
    }
    return d;
}

void build_cells(Cells &cells, const std::vector<double> &x, const double cell[3], double rc, int &nx,
                 int &ny, int &nz) {
    const int n = static_cast<int>(x.size() / 3);
    double lo[3] = {1e300, 1e300, 1e300};
    double hi[3] = {-1e300, -1e300, -1e300};
    for (int i = 0; i < n; ++i) {
        for (int a = 0; a < 3; ++a) {
            lo[a] = std::min(lo[a], x[3 * i + a]);
            hi[a] = std::max(hi[a], x[3 * i + a]);
        }
    }
    int ncell[3] = {1, 1, 1};
    double origin[3] = {0, 0, 0};
    double inv[3] = {1.0 / rc, 1.0 / rc, 1.0 / rc};
    for (int a = 0; a < 3; ++a) {
        if (cell[a] > 0.0) {
            origin[a] = 0.0;
            ncell[a] = std::max(1, static_cast<int>(std::floor(cell[a] / rc)));
            inv[a] = static_cast<double>(ncell[a]) / cell[a];
        } else {
            origin[a] = lo[a];
            const double span = std::max(0.0, hi[a] - lo[a]);
            ncell[a] = static_cast<int>(std::floor(span / rc)) + 1;
            inv[a] = 1.0 / rc;
        }
    }
    nx = ncell[0];
    ny = ncell[1];
    nz = ncell[2];
    const int ntot = nx * ny * nz;
    cells.head.assign(static_cast<size_t>(ntot), -1);
    cells.next.assign(static_cast<size_t>(n), -1);
    for (int i = 0; i < n; ++i) {
        int idx[3];
        for (int a = 0; a < 3; ++a) {
            const double t = (x[3 * i + a] - origin[a]) * inv[a];
            int k = static_cast<int>(std::floor(t));
            if (cell[a] > 0.0) {
                const int m = ncell[a];
                k %= m;
                if (k < 0) {
                    k += m;
                }
            } else if (k < 0) {
                k = 0;
            } else if (k >= ncell[a]) {
                k = ncell[a] - 1;
            }
            idx[a] = k;
        }
        const int slot = (idx[2] * ny + idx[1]) * nx + idx[0];
        cells.next[static_cast<size_t>(i)] = cells.head[static_cast<size_t>(slot)];
        cells.head[static_cast<size_t>(slot)] = i;
    }
}

struct Pot : Fg {
    bool morse = false;
    double cell[3] = {0, 0, 0};
    double cutoff = 0;
    Cells cells;
    void eval(const std::vector<double> &x, double &e, std::vector<double> &g) override {
        e = 0.0;
        g.assign(x.size(), 0.0);
        int nx = 1, ny = 1, nz = 1;
        build_cells(cells, x, cell, cutoff, nx, ny, nz);
        const double rc2 = cutoff * cutoff;
        const double d0 = 0.7102, alpha = 1.6047, r0 = 2.897;
        const double xc = std::exp(-alpha * (7.0 - r0));
        const double shift_m = d0 * (xc * xc - 2.0 * xc);
        const double s6c = std::pow(1.0 / rc2, 3.0);
        const double shift_l = 4.0 * (s6c * s6c - s6c);
        for (int iz = 0; iz < nz; ++iz) {
            for (int iy = 0; iy < ny; ++iy) {
                for (int ix = 0; ix < nx; ++ix) {
                    const int home = (iz * ny + iy) * nx + ix;
                    for (int i = cells.head[static_cast<size_t>(home)]; i >= 0;
                         i = cells.next[static_cast<size_t>(i)]) {
                        for (int dz = -1; dz <= 1; ++dz) {
                            for (int dy = -1; dy <= 1; ++dy) {
                                for (int dx = -1; dx <= 1; ++dx) {
                                    int jx = ix + dx, jy = iy + dy, jz = iz + dz;
                                    if (cell[0] > 0.0) {
                                        jx %= nx;
                                        if (jx < 0) jx += nx;
                                    } else if (jx < 0 || jx >= nx) {
                                        continue;
                                    }
                                    if (cell[1] > 0.0) {
                                        jy %= ny;
                                        if (jy < 0) jy += ny;
                                    } else if (jy < 0 || jy >= ny) {
                                        continue;
                                    }
                                    if (cell[2] > 0.0) {
                                        jz %= nz;
                                        if (jz < 0) jz += nz;
                                    } else if (jz < 0 || jz >= nz) {
                                        continue;
                                    }
                                    const int slot = (jz * ny + jy) * nx + jx;
                                    for (int j = cells.head[static_cast<size_t>(slot)]; j >= 0;
                                         j = cells.next[static_cast<size_t>(j)]) {
                                        if (j <= i) {
                                            continue;
                                        }
                                        double d[3], r2 = 0.0;
                                        for (int a = 0; a < 3; ++a) {
                                            d[a] = mic(x[3 * i + a] - x[3 * j + a], cell[a]);
                                            r2 += d[a] * d[a];
                                        }
                                        if (!(r2 < rc2 && r2 > 1.0e-16)) {
                                            continue;
                                        }
                                        double du = 0.0;
                                        if (!morse) {
                                            const double s6 = std::pow(r2, -3.0);
                                            e += 4.0 * (s6 * s6 - s6) - shift_l;
                                            du = 24.0 * (s6 - 2.0 * s6 * s6) / r2;
                                        } else {
                                            const double r = std::sqrt(r2);
                                            if (r < 1.0e-8) {
                                                continue;
                                            }
                                            const double ex = std::exp(-alpha * (r - r0));
                                            e += d0 * (ex * ex - 2.0 * ex) - shift_m;
                                            du = 2.0 * d0 * alpha * ex * (1.0 - ex) / r;
                                        }
                                        for (int a = 0; a < 3; ++a) {
                                            const double c = du * d[a];
                                            g[3 * i + a] += c;
                                            g[3 * j + a] -= c;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

std::vector<double> fcc_chunk(int atoms, double a, double rattle) {
    const int cells = std::max(1, static_cast<int>(std::ceil(std::cbrt(atoms / 4.0))));
    const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
    std::vector<std::array<double, 3>> raw;
    for (int iz = 0; iz < cells; ++iz) {
        for (int iy = 0; iy < cells; ++iy) {
            for (int ix = 0; ix < cells; ++ix) {
                for (const auto &b : basis) {
                    raw.push_back({(ix + b[0]) * a, (iy + b[1]) * a, (iz + b[2]) * a});
                }
            }
        }
    }
    raw.resize(static_cast<size_t>(atoms));
    std::vector<double> x;
    x.reserve(static_cast<size_t>(3 * atoms));
    for (int k = 0; k < atoms; ++k) {
        for (int c = 0; c < 3; ++c) {
            x.push_back(raw[static_cast<size_t>(k)][c] + rattle * std::sin((3 * k + c) * 1.7));
        }
    }
    return x;
}

std::vector<double> fcc_slab(int nx, int ny, int nz, double a, double rattle, double cell[3]) {
    const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
    std::vector<double> x;
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                for (const auto &b : basis) {
                    const double p[3] = {(ix + b[0]) * a, (iy + b[1]) * a, (iz + b[2]) * a};
                    const int k = static_cast<int>(x.size() / 3);
                    for (int c = 0; c < 3; ++c) {
                        x.push_back(p[c] + rattle * std::sin((3 * k + c) * 1.7));
                    }
                }
            }
        }
    }
    cell[0] = nx * a;
    cell[1] = ny * a;
    cell[2] = 0.0;
    return x;
}

void slab_counts(int atoms, int &nx, int &ny, int &nz) {
    if (atoms == 500) {
        nx = ny = nz = 5;
    } else if (atoms == 5000) {
        nx = 25;
        ny = 10;
        nz = 5;
    } else {
        nx = 50;
        ny = 25;
        nz = 10;
    }
}

System make_system(bool morse, bool slab, int atoms) {
    System s;
    s.morse = morse;
    s.atoms = atoms;
    const double lj_nn = 1.122462048309373;
    const double spacing = morse ? 2.897 : lj_nn;
    const double a = spacing * std::sqrt(2.0);
    s.cutoff = morse ? 7.0 : 2.5;
    if (!slab) {
        s.x0 = fcc_chunk(atoms, a, 0.02);
        s.name = std::string(morse ? "morse" : "lj") + "-cluster-" + std::to_string(atoms);
    } else {
        int nx, ny, nz;
        slab_counts(atoms, nx, ny, nz);
        s.x0 = fcc_slab(nx, ny, nz, a, 0.02, s.cell);
        s.name = std::string(morse ? "morse" : "lj") + "-slab-" + std::to_string(atoms);
    }
    return s;
}

}  // namespace

int main(int argc, char **argv) {
    const std::string out_path = argc > 1 ? argv[1] : "bench/raw/large_n_eon.csv";
    std::ofstream out(out_path);
    out << "system,potential,geometry,atoms,arm,force_calls,fmax,energy,converged\n";
    const int sizes[] = {500, 5000, 50000};
    const char *arms[] = {"lbfgs", "fire"};
    for (int atoms : sizes) {
        for (int morse = 0; morse < 2; ++morse) {
            for (int slab = 0; slab < 2; ++slab) {
                System sys = make_system(morse != 0, slab != 0, atoms);
                for (const char *arm : arms) {
                    Pot pot;
                    pot.morse = sys.morse;
                    pot.cutoff = sys.cutoff;
                    pot.cell[0] = sys.cell[0];
                    pot.cell[1] = sys.cell[1];
                    pot.cell[2] = sys.cell[2];
                    double e0 = 0;
                    std::vector<double> g0(sys.x0.size());
                    pot.eval(sys.x0, e0, g0);
                    std::cerr << sys.name << " E/atom " << e0 / atoms << " fmax0 " << stop_grad(g0, true)
                              << "\n";
                    Counters c;
                    std::vector<double> x = sys.x0;
                    double e = 0;
                    std::vector<double> g(x.size());
                    call(pot, c, x, e, g);
                    bool converged = false;
                    if (std::string(arm) == "lbfgs") {
                        Lbfgs opt;
                        for (int it = 0; it < 400; ++it) {
                            if (stop_grad(g, true) < 0.01) {
                                converged = true;
                                break;
                            }
                            opt.step(pot, c, x, e, g, kMaxMove);
                        }
                    } else {
                        Fire opt;
                        for (int it = 0; it < 400 && !opt.stalled; ++it) {
                            if (stop_grad(g, true) < 0.01) {
                                converged = true;
                                break;
                            }
                            opt.step(pot, c, x, g, kMaxMove);
                        }
                    }
                    double ef = 0;
                    std::vector<double> gf(x.size());
                    pot.eval(x, ef, gf);
                    const double fmax = stop_grad(gf, true);
                    if (fmax < 0.01) {
                        converged = true;
                    }
                    const char *pot_name = sys.morse ? "morse" : "lj";
                    const char *geom = sys.name.find("slab") != std::string::npos ? "slab" : "cluster";
                    out << sys.name << "," << pot_name << "," << geom << "," << atoms << ",eon-" << arm << ","
                        << c.evals << "," << std::scientific << fmax << "," << ef << ","
                        << (converged ? "1" : "0") << "\n";
                    out.flush();
                    std::cerr << sys.name << " eon-" << arm << " calls " << c.evals << " fmax " << fmax
                              << " conv " << converged << "\n";
                }
            }
        }
    }
    std::cerr << "wrote " << out_path << "\n";
    return 0;
}
