import numpy as np
from collections import namedtuple
from matipo import sequence as seq
from matipo import ParDef
from matipo import datalayout
import logging

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

def float_array(v):
    a = np.array(v, dtype=float)
    a.setflags(write=False)
    return a

g_ZERO = float_array((0, 0, 0))

PARDEF = [
    ParDef('n_scans', int, 1, min=1),
    ParDef('f', float, 1e6),
    ParDef('a_pulse', float, 0.25),
    ParDef('t_pulse', float, 32e-6),
    ParDef('n_leading_pulses', int, 0),
    ParDef('t_dw', float, 1e-6),
    ParDef('n_samples', int, 1000),
    ParDef('t_read', float, 0.001),
    ParDef('g_read', float_array, (0, 0, 1), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_spoil', float, 100e-6),
    ParDef('g_spoil', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_phase', float, 0),
    ParDef('g_phase_read', float_array, (0, 0, -1), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('n_phase_1', int, 1, min=1),
    ParDef('g_phase_1', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('n_phase_2', int, 1, min=1),
    ParDef('g_phase_2', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_grad_stab', float, 100e-6),
    ParDef('t_end', float, 0.5),
    ParDef('shim_x', float, 0, min=-1, max=1),
    ParDef('shim_y', float, 0, min=-1, max=1),
    ParDef('shim_z', float, 0, min=-1, max=1),
    ParDef('shim_z2', float, 0, min=-1, max=1),
    ParDef('shim_zx', float, 0, min=-1, max=1),
    ParDef('shim_zy', float, 0, min=-1, max=1),
    ParDef('shim_xy', float, 0, min=-1, max=1),
    ParDef('shim_x2y2', float, 0, min=-1, max=1),
    ParDef('g_venc', float_array, (0, 0, 0.3), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_venc', float, 1000e-6),
]

ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])

class Sequence(seq.Sequence):
    def __init__(self):
        super().__init__()
        self.pardef = PARDEF

    def get_options(self):
        # Instance method - runtime options
        return seq.Options(
            amp_enabled=True,
            rx_gain=7,
        )

    def get_datalayout(self):
        par = self.par
        return datalayout.Scans(
            par.n_scans,
            datalayout.Repetitions(
                par.n_phase_1,
                datalayout.Repetitions(
                    par.n_phase_2,
                    datalayout.Acquisition(
                        n_samples=par.n_samples,
                        t_dw=par.t_dw
                    )
                )
            )
        )

    def main(self):
        par = self.par

        print("Running FISP_EDIT sequence with parameters:")
        print(par)

        none_params = [name for name in par._fields if getattr(par, name) is None]
        if none_params:
            raise Exception(f"Parameters with None values: {none_params}")

        t_rep = par.t_pulse + 2*par.t_phase + par.t_grad_stab + par.t_read + par.t_spoil + par.t_end

        grad_total_area = (
            (np.abs(par.g_phase_read) + np.abs(par.g_phase_1) + np.abs(par.g_phase_2)) * par.t_phase
            + np.abs(par.g_read) * par.t_read
            + np.abs(par.g_spoil) * par.t_spoil
        )
        grad_duty_cycle = grad_total_area / t_rep
        log.debug(f'Gradient duty cycle: {grad_duty_cycle}')
        if np.any(grad_duty_cycle > 0.3):
            raise Exception('Gradient duty cycle too high!')

        g_phase_1_step = -par.g_phase_1 / (par.n_phase_1 // 2) if par.n_phase_1 > 1 else 0
        g_phase_2_step = -par.g_phase_2 / (par.n_phase_2 // 2) if par.n_phase_2 > 1 else 0

        if par.t_read < par.n_samples * par.t_dw:
            raise Exception('Read gradient time too short for acquisition!')

        t_evo = par.t_pulse / 2 + par.t_phase + par.t_grad_stab + par.t_read / 2
        log.debug(f"Evolution time: {t_evo}")

        rf_pulse_even = (
            seq.pulse_start(par.f, 0, par.a_pulse)
            + seq.wait(par.t_pulse)
            + seq.pulse_end()
        )

        rf_pulse_odd = (
            seq.pulse_start(par.f, 180, par.a_pulse)
            + seq.wait(par.t_pulse)
            + seq.pulse_end()
        )

        readout_even = (
            seq.gradient(*par.g_read)
            + seq.acquire(par.f, 0, par.t_dw, par.n_samples)
            + seq.wait(par.t_read)
        )
        readout_odd = (
            seq.gradient(*par.g_read)
            + seq.acquire(par.f, 180, par.t_dw, par.n_samples)
            + seq.wait(par.t_read)
        )

        phase = 0

        yield seq.shim(
            par.shim_x, par.shim_y, par.shim_z,
            par.shim_z2, par.shim_zx, par.shim_zy,
            par.shim_xy, par.shim_x2y2
        )
        yield seq.wait(0.01)

        for _ in range(par.n_leading_pulses):
            phase = (phase + 180) % 360
            yield seq.pulse_start(par.f, phase, par.a_pulse)
            yield seq.wait(par.t_pulse)
            yield seq.pulse_end()
            yield seq.wait(par.t_phase)
            yield seq.wait(par.t_grad_stab)
            yield seq.wait(par.t_read)
            yield seq.wait(par.t_phase)
            yield seq.gradient(*par.g_spoil)
            yield seq.wait(par.t_spoil)
            yield seq.gradient(*g_ZERO)
            yield seq.wait(par.t_end)

        i = 0
        for venc_sign in [+1, -1]:
            for i_scan in range(par.n_scans):
                for i_phase_1 in range(par.n_phase_1):
                    g_phase_1_i = par.g_phase_1 + i_phase_1 * g_phase_1_step
                    for i_phase_2 in range(par.n_phase_2):
                        g_phase_2_i = par.g_phase_2 + i_phase_2 * g_phase_2_step

                        even = i % 2 == 0
                        g_venc_signed = tuple(venc_sign * g for g in par.g_venc)

                        yield (
                            (rf_pulse_even if even else rf_pulse_odd)
                            + seq.gradient(*(par.g_phase_read + g_phase_1_i + g_phase_2_i))
                            + seq.wait(par.t_phase)
                            + seq.gradient(*g_ZERO)
                            + seq.wait(par.t_grad_stab)
                            + seq.gradient(*g_venc_signed)
                            + seq.wait(par.t_venc)
                            + seq.gradient(*tuple(-x for x in g_venc_signed))
                            + seq.wait(par.t_venc)
                            + (readout_even if even else readout_odd)
                            + seq.gradient(*(-g_phase_1_i - g_phase_2_i))
                            + seq.wait(par.t_phase)
                            + seq.gradient(*par.g_spoil)
                            + seq.wait(par.t_spoil)
                            + seq.gradient(*g_ZERO)
                            + seq.wait(par.t_end)
                        )
                        i += 1


def get_options(_=None):
    return [
        {
            'name': 'g_venc',
            'type': 'float',
            'default': 0.3,
            'label': 'Velocity encoding gradient amplitude (T/m)'
        },
    ]


