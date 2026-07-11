#include "Heat.hpp"
#include "prm_parse.hpp"

#include <deal.II/base/exceptions.h>

#include <cmath>
#include <filesystem>
#include <string>

// Main function.
int main(int argc, char *argv[])
{
	constexpr unsigned int dim = Heat::dim;

	Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

	//AssertThrow(argc <= 2, ExcMessage("Usage: ./space_time_adaptivity [parameter-file.prm]"));

	//const std::string parameter_file = (argc == 2) ? argv[1] : "../config/default.prm";
	const std::string parameter_file = "../config/default.prm";
	const Parameters  parameters     = read_project_parameters(parameter_file);

	const auto mu = [parameters](const Point<dim> & /*p*/) { return parameters.diffusion.mu; };

	const auto f = [parameters](const Point<dim> &p, const double &t)
	{
		const double g =
		    std::exp(-parameters.source.a * std::cos(2.0 * numbers::PI * parameters.source.N * t)) /
		    std::exp(parameters.source.a);

		const double r2 =
		    (p[0] - parameters.source.center_x) * (p[0] - parameters.source.center_x) +
		    (p[1] - parameters.source.center_y) * (p[1] - parameters.source.center_y) +
		    (p[2] - parameters.source.center_z) * (p[2] - parameters.source.center_z);

		const double h = std::exp(-r2 / (parameters.source.sigma * parameters.source.sigma));

		return g * h;
	};

	Heat problem(parameters.mesh.file_name,
	             parameters.mesh.degree,
	             parameters.time.final_time,
	             parameters.time.theta,
	             parameters.time.initial_delta_t,
	             mu,
	             f,
	             parameters.output.directory,
	             parameters.output.basename);

	problem.run();

	return 0;
}