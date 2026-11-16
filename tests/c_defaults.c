/* Every caller-filled record, started from its initializer, holds the Rust
 * default in every field. The harness passes each record as NAME:SIZE and
 * each field as NAME:OFFSET:SIZE:BITS from the Rust structs. A field that
 * one side names and the other does not fails, as does any size, offset or
 * bit pattern that differs. Each record is built where the stack held 0x40
 * bytes: a double nobody wrote reads 32.5, a positive noise_eps, which
 * turns the noise-floor stop on. A minimize and a session from the default
 * control then converge on a unit quadratic, and SCG reports the same from
 * the default records as from NULL. */

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "rgmin.h"

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

#define POISON 0x40
#define COUNT(a) (sizeof(a) / sizeof((a)[0]))

typedef struct field_t {
    const char *name;
    size_t offset;
    size_t size;
} field_t;

#define FIELD(type, member) {#type "." #member, offsetof(type, member), sizeof(((type *)0)->member)}
#define ELEMENT(type, member, i)                                                                  \
    {#type "." #member "[" #i "]", offsetof(type, member) + (i) * sizeof(((type *)0)->member[0]), \
     sizeof(((type *)0)->member[0])}

static const field_t control_fields[] = {
    FIELD(rgmin_control_t, maxiter),
    FIELD(rgmin_control_t, gtol),
    FIELD(rgmin_control_t, istep),
    FIELD(rgmin_control_t, memory),
    FIELD(rgmin_control_t, maxmove),
    FIELD(rgmin_control_t, noise_eps),
    FIELD(rgmin_control_t, noise_c),
};

static const field_t exp_fields[] = {
    FIELD(rgmin_exp_t, a),
    FIELD(rgmin_exp_t, c_stab),
    FIELD(rgmin_exp_t, mu),
    FIELD(rgmin_exp_t, r_nn),
    FIELD(rgmin_exp_t, r_cut),
    FIELD(rgmin_exp_t, length_x),
    FIELD(rgmin_exp_t, length_y),
    FIELD(rgmin_exp_t, length_z),
    FIELD(rgmin_exp_t, recalc_mu),
};

static const field_t scg_params_fields[] = {
    FIELD(rgmin_scg_params_t, sigma0),
    FIELD(rgmin_scg_params_t, lambda),
    FIELD(rgmin_scg_params_t, lambda_limit),
    FIELD(rgmin_scg_params_t, tol_sol),
    FIELD(rgmin_scg_params_t, tol_func),
    FIELD(rgmin_scg_params_t, conjugacy),
};

static const field_t scg_options_fields[] = {
    FIELD(rgmin_scg_options_t, step_tolerance),
};

static const field_t eigen_params_fields[] = {
    FIELD(rgmin_eigen_params_t, kind),
    FIELD(rgmin_eigen_params_t, nev),
    FIELD(rgmin_eigen_params_t, krylov),
    FIELD(rgmin_eigen_params_t, max_iter),
    FIELD(rgmin_eigen_params_t, tol),
};

static const field_t eigen_options_fields[] = {
    FIELD(rgmin_eigen_options_t, degree),
    FIELD(rgmin_eigen_options_t, extra),
};

static const field_t precon_fields[] = {
    FIELD(rgmin_precon_t, struct_size),
    FIELD(rgmin_precon_t, kind),
    FIELD(rgmin_precon_t, c_stab),
    FIELD(rgmin_precon_t, mu),
    FIELD(rgmin_precon_t, r_cut),
    ELEMENT(rgmin_precon_t, cell, 0),
    ELEMENT(rgmin_precon_t, cell, 1),
    ELEMENT(rgmin_precon_t, cell, 2),
    FIELD(rgmin_precon_t, numbers),
    FIELD(rgmin_precon_t, natoms),
    FIELD(rgmin_precon_t, ghost_z),
    FIELD(rgmin_precon_t, ghost_pos),
    FIELD(rgmin_precon_t, nghost),
    FIELD(rgmin_precon_t, nbr_i),
    FIELD(rgmin_precon_t, nbr_j),
    FIELD(rgmin_precon_t, nbr_r),
    FIELD(rgmin_precon_t, npairs),
    FIELD(rgmin_precon_t, anchor_i),
    FIELD(rgmin_precon_t, anchor_r),
    FIELD(rgmin_precon_t, nanchor),
    FIELD(rgmin_precon_t, exp_a),
    FIELD(rgmin_precon_t, bonds),
    FIELD(rgmin_precon_t, nbonds),
    FIELD(rgmin_precon_t, angles),
    FIELD(rgmin_precon_t, nangles),
    FIELD(rgmin_precon_t, dihedrals),
    FIELD(rgmin_precon_t, geometry_kind),
};

typedef struct expected_t {
    char name[96];
    int record;
    size_t offset;
    size_t size;
    uint64_t bits;
    int used;
} expected_t;

static expected_t expected[160];
static size_t n_expected;

static int parse(int argc, char **argv) {
    int i;
    for (i = 1; i < argc; ++i) {
        expected_t *e;
        int got;
        if (n_expected == COUNT(expected)) {
            fprintf(stderr, "more than %zu expected values\n", COUNT(expected));
            return 1;
        }
        e = &expected[n_expected++];
        memset(e, 0, sizeof *e);
        got = sscanf(argv[i], "%95[^:]:%zu:%zu:%" SCNx64, e->name, &e->offset, &e->size, &e->bits);
        if (got == 2) {
            e->record = 1;
            e->size = e->offset;
            e->offset = 0;
        } else if (got != 4) {
            fprintf(stderr, "cannot parse %s\n", argv[i]);
            return 1;
        }
    }
    return 0;
}

static expected_t *find(const char *name, int record) {
    size_t i;
    for (i = 0; i < n_expected; ++i) {
        if (expected[i].record == record && strcmp(expected[i].name, name) == 0) {
            expected[i].used = 1;
            return &expected[i];
        }
    }
    return NULL;
}

static int read_bits(const unsigned char *at, size_t size, uint64_t *bits) {
    if (size == sizeof(uint64_t)) {
        memcpy(bits, at, size);
        return 1;
    }
    if (size == sizeof(uint32_t)) {
        uint32_t narrow;
        memcpy(&narrow, at, size);
        *bits = narrow;
        return 1;
    }
    return 0;
}

static int check(const char *record, const void *value, size_t size, const field_t *fields,
                 size_t count) {
    int failed = 0;
    size_t i;
    const expected_t *e = find(record, 1);
    if (e == NULL) {
        fprintf(stderr, "%s: the harness sends no size\n", record);
        failed = 1;
    } else if (e->size != size) {
        fprintf(stderr, "%s: %zu bytes here, %zu in Rust\n", record, size, e->size);
        failed = 1;
    }
    for (i = 0; i < count; ++i) {
        const field_t *f = &fields[i];
        uint64_t bits = 0;
        e = find(f->name, 0);
        if (e == NULL) {
            fprintf(stderr, "%s: the harness sends no default\n", f->name);
            failed = 1;
        } else if (e->offset != f->offset || e->size != f->size) {
            fprintf(stderr, "%s: offset %zu size %zu here, offset %zu size %zu in Rust\n", f->name,
                    f->offset, f->size, e->offset, e->size);
            failed = 1;
        } else if (!read_bits((const unsigned char *)value + f->offset, f->size, &bits)) {
            fprintf(stderr, "%s: no reader for %zu bytes\n", f->name, f->size);
            failed = 1;
        } else if (bits != e->bits) {
            fprintf(stderr, "%s: bits %016" PRIx64 ", Rust default %016" PRIx64 "\n", f->name,
                    bits, e->bits);
            failed = 1;
        }
    }
    return failed;
}

static NOINLINE void poison_stack(void) {
    volatile unsigned char bytes[1 << 16];
    size_t i;
    for (i = 0; i < sizeof bytes; ++i) {
        bytes[i] = POISON;
    }
}

static NOINLINE int check_control(void) {
    rgmin_control_t c = rgmin_control_default();
    return check("rgmin_control_t", &c, sizeof c, control_fields, COUNT(control_fields));
}

static NOINLINE int check_exp(void) {
    rgmin_exp_t c = rgmin_exp_default();
    return check("rgmin_exp_t", &c, sizeof c, exp_fields, COUNT(exp_fields));
}

static NOINLINE int check_scg_params(void) {
    rgmin_scg_params_t c = rgmin_scg_params_default();
    return check("rgmin_scg_params_t", &c, sizeof c, scg_params_fields,
                 COUNT(scg_params_fields));
}

static NOINLINE int check_scg_options(void) {
    rgmin_scg_options_t c = rgmin_scg_options_default();
    return check("rgmin_scg_options_t", &c, sizeof c, scg_options_fields,
                 COUNT(scg_options_fields));
}

static NOINLINE int check_eigen_params(void) {
    rgmin_eigen_params_t c = rgmin_eigen_params_default();
    return check("rgmin_eigen_params_t", &c, sizeof c, eigen_params_fields,
                 COUNT(eigen_params_fields));
}

static NOINLINE int check_eigen_options(void) {
    rgmin_eigen_options_t c = rgmin_eigen_options_default();
    return check("rgmin_eigen_options_t", &c, sizeof c, eigen_options_fields,
                 COUNT(eigen_options_fields));
}

static NOINLINE int check_precon(void) {
    rgmin_precon_t c;
    memset(&c, POISON, sizeof c);
    rgmin_precon_init(&c, RGMIN_PRECON_EXP);
    return check("rgmin_precon_t", &c, sizeof c, precon_fields, COUNT(precon_fields));
}

static rgmin_status_t bowl_eval(void *user, const DLManagedTensorVersioned *x, double *value_out) {
    const double *p = (const double *)x->dl_tensor.data;
    (void)user;
    *value_out = 0.5 * p[0] * p[0];
    return RGMIN_SUCCESS;
}

static rgmin_status_t bowl_grad(void *user, const DLManagedTensorVersioned *x,
                                DLManagedTensorVersioned *grad_out) {
    const double *p = (const double *)x->dl_tensor.data;
    double *g = (double *)grad_out->dl_tensor.data;
    (void)user;
    g[0] = p[0];
    return RGMIN_SUCCESS;
}

static rgmin_status_t bowl_fg(void *user, const DLManagedTensorVersioned *x, double *value_out,
                              DLManagedTensorVersioned *grad_out) {
    bowl_eval(user, x, value_out);
    return bowl_grad(user, x, grad_out);
}

/* |g| = 0.02 at the start sits under any floor a stray noise_eps sets. */
static NOINLINE int minimize_from_default(void) {
    double x = 0.02;
    rgmin_control_t ctrl = rgmin_control_default();
    rgmin_report_t out = {.stop = RGMIN_STOP_BUDGET};
    rgmin_status_t st;
    DLManagedTensorVersioned *tensor = rgmin_tensor_borrow_cpu_f64(&x, 1);
    if (tensor == NULL) {
        return 1;
    }
    st = rgmin_minimize(bowl_eval, bowl_grad, NULL, tensor, &ctrl, RGMIN_LBFGS, &out);
    rgmin_tensor_free(tensor);
    if (st != RGMIN_SUCCESS || out.stop != RGMIN_STOP_TOLERANCE) {
        fprintf(stderr, "minimize: status %d stop %d grad %g steps %zu\n", (int)st, (int)out.stop,
                out.grad_norm, out.steps);
        return 1;
    }
    return 0;
}

static NOINLINE int session_from_default(void) {
    double x = 0.02;
    rgmin_control_t ctrl = rgmin_control_default();
    rgmin_report_t out = {.stop = RGMIN_STOP_BUDGET};
    rgmin_solver_t *s = rgmin_solver_create(RGMIN_LBFGS, &ctrl, 1);
    DLManagedTensorVersioned *tensor = rgmin_tensor_borrow_cpu_f64(&x, 1);
    int failed = 1;
    int step;
    for (step = 0; s != NULL && tensor != NULL && step < 20; ++step) {
        if (rgmin_solver_step_fg(s, bowl_fg, NULL, tensor, &out) != RGMIN_SUCCESS ||
            out.stop == RGMIN_STOP_NOISE_FLOOR) {
            break;
        }
        if (out.grad_norm < ctrl.gtol) {
            failed = 0;
            break;
        }
    }
    if (failed) {
        fprintf(stderr, "session: stop %d grad %g x %g after %d steps\n", (int)out.stop,
                out.grad_norm, x, step);
    }
    rgmin_tensor_free(tensor);
    rgmin_solver_free(s);
    return failed;
}

static int scg(const rgmin_scg_params_t *params, const rgmin_scg_options_t *options,
               rgmin_report_t *out, double *x_out) {
    double x = 1.5;
    rgmin_control_t ctrl = rgmin_control_default();
    rgmin_status_t st;
    DLManagedTensorVersioned *tensor = rgmin_tensor_borrow_cpu_f64(&x, 1);
    if (tensor == NULL) {
        return 1;
    }
    out->stop = RGMIN_STOP_BUDGET;
    st = rgmin_minimize_scg_with_options(bowl_eval, bowl_grad, NULL, NULL, tensor, &ctrl, params,
                                         options, out);
    rgmin_tensor_free(tensor);
    *x_out = x;
    if (st != RGMIN_SUCCESS) {
        fprintf(stderr, "scg: %s\n", rgmin_last_error());
        return 1;
    }
    return 0;
}

static NOINLINE int scg_defaults_are_null(void) {
    rgmin_scg_params_t params = rgmin_scg_params_default();
    rgmin_scg_options_t options = rgmin_scg_options_default();
    rgmin_report_t from_null = {.stop = RGMIN_STOP_BUDGET};
    rgmin_report_t from_default = {.stop = RGMIN_STOP_BUDGET};
    double x_null = 0.0;
    double x_default = 0.0;
    if (scg(NULL, NULL, &from_null, &x_null) != 0 ||
        scg(&params, &options, &from_default, &x_default) != 0) {
        return 1;
    }
    if (memcmp(&x_null, &x_default, sizeof x_null) != 0 ||
        memcmp(&from_null.value, &from_default.value, sizeof from_null.value) != 0 ||
        memcmp(&from_null.grad_norm, &from_default.grad_norm, sizeof from_null.grad_norm) != 0 ||
        from_null.steps != from_default.steps || from_null.stop != from_default.stop) {
        fprintf(stderr, "scg: NULL gives x %g steps %zu stop %d, defaults x %g steps %zu stop %d\n",
                x_null, from_null.steps, (int)from_null.stop, x_default, from_default.steps,
                (int)from_default.stop);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    int failed = parse(argc, argv);
    size_t i;
    poison_stack();
    failed |= check_control();
    poison_stack();
    failed |= check_exp();
    poison_stack();
    failed |= check_scg_params();
    poison_stack();
    failed |= check_scg_options();
    poison_stack();
    failed |= check_eigen_params();
    poison_stack();
    failed |= check_eigen_options();
    poison_stack();
    failed |= check_precon();
    for (i = 0; i < n_expected; ++i) {
        if (!expected[i].used) {
            fprintf(stderr, "%s: Rust sends it and nothing here checks it\n", expected[i].name);
            failed = 1;
        }
    }
    poison_stack();
    failed |= minimize_from_default();
    poison_stack();
    failed |= session_from_default();
    poison_stack();
    failed |= scg_defaults_are_null();
    return failed;
}
