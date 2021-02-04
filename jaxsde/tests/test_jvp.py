from jax.config import config
config.update("jax_enable_x64", True)

from jax.test_util import check_jvp

from jaxsde import ito_integrate, jvp_ito_integrate
from test_sdeint import make_example_sde


def test_ito_int_jvp():
    # forward mode
    f, g, b, y0, ts, dt = make_example_sde()

    def onearg_int(y0):
        return ito_integrate(f, g, y0, ts, b, dt)

    def odeint2_jvp((y0,), (tan_y,)):
        return jvp_ito_integrate(tan_y, y0, f, g, ts, b, dt=dt, args=())

    check_jvp(onearg_int, odeint2_jvp, (y0,), atol=1e-3, rtol=1e-3)
