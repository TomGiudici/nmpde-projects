#include "Heat_adaptive.hpp"
#include <cmath>

int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Heat::dim;
  using Point = dealii::Point<dim>;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const auto mu = [](const Point & /*p*/) {
    return 1.0;
  };

  const double a     = 2.0;
  const double N     = 3.0;
  const double x0    = 0.5;
  const double y0    = 0.5;
  const double z0    = 0.5;
  const double sigma = 0.1;

  const auto f = [a, N, x0, y0, z0, sigma](const Point &p, const double &t) {
    const double g =
      std::exp(-a * std::cos(2.0 * numbers::PI * N * t)) / std::exp(a);

    const double r2 =
      (p[0] - x0) * (p[0] - x0) +
      (p[1] - y0) * (p[1] - y0) +
      (p[2] - z0) * (p[2] - z0);

    const double h = std::exp(-r2 / (sigma * sigma));

    return g * h;
  };

  Heat problem(/* mesh_filename = */ "../mesh/mesh-cube-10.msh",
               /* degree        = */ 1,
               /* T             = */ 1.0,
               /* theta         = */ 1.0,
               /* delta_t       = */ 0.0025,
               mu,
               f);

  problem.run();

  return 0;
}