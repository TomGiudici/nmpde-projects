#include "prm_parse.hpp"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

#include <fstream>

using namespace dealii;

namespace
{

	// Declare every parameter accepted by the .prm file.
	//
	// ParameterHandler uses subsection names to mirror the layout of config/default.prm.
	// Patterns are lightweight runtime checks performed while	
	// parsing the input file.
	void declare_parameters(ParameterHandler &prm)
	{
		
		prm.enter_subsection("Mesh");
		{
			prm.declare_entry("file_name",
							"../mesh/mesh-cube-10.mesh",
							Patterns::Anything(),
							"Path to the input Gmsh mesh file.");
			prm.declare_entry("degree", "1", Patterns::Integer(1), "Polynomial degree.");
		}
		prm.leave_subsection();

		prm.enter_subsection("Time");
		{
			prm.declare_entry("final_time", "1.0", Patterns::Double(0.0), "Final time T.");
			prm.declare_entry("theta",
							"1.0",
							Patterns::Double(0.0, 1.0),
							"Theta-methed parameter. Use 1.0 "
							"for Backward Euler.");
			prm.declare_entry("initial_delta_t", "0.0025", Patterns::Double(0.0), "Initial time step.");
			prm.declare_entry("min_delta_t",
							"1e-6",
							Patterns::Double(0.0),
							"Lower bound for future adaptive "
							"time stepping.");
			prm.declare_entry("max_delta_t",
							"1e-2",
							Patterns::Double(0.0),
							"Upper bound for future adaptive "
							"time stepping.");
		}
		prm.leave_subsection();

		prm.enter_subsection("Diffusion");
		{
			prm.declare_entry("mu", "1.0", Patterns::Double(0.0), "Diffusion coefficient.");
		}
		prm.leave_subsection();

		prm.enter_subsection("Source");
		{
			prm.declare_entry("a", "5.0", Patterns::Double(0.0), "Sharpness of g(t).");
			prm.declare_entry("N", "5", Patterns::Integer(1), "Number of temporal pulses.");
			prm.declare_entry("sigma", "0.1", Patterns::Double(0.0), "Spatial Gaussian width.");
			prm.declare_entry("center_x", "0.5", Patterns::Double(), "Source center x.");
			prm.declare_entry("center_y", "0.5", Patterns::Double(), "Source center y.");
			prm.declare_entry("center_z", "0.5", Patterns::Double(), "Source center z.");
		}
		prm.leave_subsection();

		prm.enter_subsection("Output");
		{
			prm.declare_entry("directory", "output", Patterns::Anything(), "Output directory.");
			prm.declare_entry("basename", "heat", Patterns::Anything(), "Output basename.");
		}
		prm.leave_subsection();

		prm.enter_subsection("Adaptivity");
		{
			prm.declare_entry("mode",
							"none",
							Patterns::Selection("none|time|space|space_time"),
							"Adaptivity mode planned for the project.");
			prm.declare_entry("time_tolerance_high",
							"1e-4",
							Patterns::Double(0.0),
							"Reject a future adaptive step above this indicator.");
			prm.declare_entry("time_tolerance_low",
							"2e-6",
							Patterns::Double(0.0),
							"Grow a future adaptive step below this indicator.");
			prm.declare_entry("refine_every_n_steps",
							"5",
							Patterns::Integer(1),
							"Accepted time steps between spatial adaptation calls.");
			prm.declare_entry("max_refinement_level",
							"8",
							Patterns::Integer(1),
							"Maximum mesh refinement level for spatial adaptivity.");
			prm.declare_entry("space_refine_fraction",
							"0.3",
							Patterns::Double(0.0, 1.0),
							"Fraction of cells marked for refinement.");
			prm.declare_entry("space_coarsen_fraction",
							"0.1",
							Patterns::Double(0.0, 1.0),
							"Fraction of cells marked for coarsening.");
		}
		prm.leave_subsection();
	}

	// Validate relationships between parameters after parsing.
	void validate(const Parameters &parameters)
	{
		AssertThrow(parameters.time.final_time > 0.0, ExcMessage("Time/final_time must be positive."));
		AssertThrow(parameters.time.initial_delta_t > 0.0,
					ExcMessage("Time/initial_delta_t must be positive."));
		AssertThrow(parameters.time.min_delta_t > 0.0,
					ExcMessage("Time/min_delta_t must be positive."));
		AssertThrow(parameters.time.max_delta_t >= parameters.time.min_delta_t,
					ExcMessage("Time/max_delta_t must be at least Time/min_delta_t."));
		AssertThrow(parameters.time.initial_delta_t >= parameters.time.min_delta_t &&
						parameters.time.initial_delta_t <= parameters.time.max_delta_t,
					ExcMessage("Time/initial_delta_t must lie in [min_delta_t, max_delta_t]."));
		AssertThrow(parameters.time.theta >= 0.5 && parameters.time.theta <= 1.0,
					ExcMessage("Use Time/theta in [0.5, 1.0] for stable diffusion solves."));
		AssertThrow(parameters.diffusion.mu > 0.0, ExcMessage("Diffusion/mu must be positive."));
		AssertThrow(parameters.source.sigma > 0.0, ExcMessage("Source/sigma must be positive."));
		AssertThrow(parameters.adaptivity.time_tolerance_high >=
						parameters.adaptivity.time_tolerance_low,
					ExcMessage("Adaptivity/time_tolerance_high must be at least "
							"Adaptivity/time_tolerance_low."));
		AssertThrow(parameters.adaptivity.space_refine_fraction >= 0.0 &&
						parameters.adaptivity.space_refine_fraction <= 1.0,
					ExcMessage("Adaptivity/space_refine_fraction must be in [0, 1]."));
		AssertThrow(parameters.adaptivity.space_coarsen_fraction >= 0.0 &&
						parameters.adaptivity.space_coarsen_fraction <= 1.0,
					ExcMessage("Adaptivity/space_coarsen_fraction must be in [0, 1]."));
}
} // namespace

Parameters read_project_parameters(const std::string &file_name)
{
	ParameterHandler prm;
	declare_parameters(prm);
	prm.parse_input(file_name);

	Parameters parameters;

	// Convert ParameterHandler's string storage into the typed project struct.
	prm.enter_subsection("Mesh");
	{
		parameters.mesh.file_name = prm.get("file_name");
		parameters.mesh.degree    = prm.get_integer("degree");
	}
	prm.leave_subsection();

	prm.enter_subsection("Time");
	{
		parameters.time.final_time      = prm.get_double("final_time");
		parameters.time.theta           = prm.get_double("theta");
		parameters.time.initial_delta_t = prm.get_double("initial_delta_t");
		parameters.time.min_delta_t     = prm.get_double("min_delta_t");
		parameters.time.max_delta_t     = prm.get_double("max_delta_t");
	}
	prm.leave_subsection();

	prm.enter_subsection("Diffusion");
	{
		parameters.diffusion.mu = prm.get_double("mu");
	}
	prm.leave_subsection();

	prm.enter_subsection("Source");
	{
		parameters.source.a        = prm.get_double("a");
		parameters.source.N        = prm.get_integer("N");
		parameters.source.sigma    = prm.get_double("sigma");
		parameters.source.center_x = prm.get_double("center_x");
		parameters.source.center_y = prm.get_double("center_y");
		parameters.source.center_z = prm.get_double("center_z");
	}
	prm.leave_subsection();

	prm.enter_subsection("Output");
	{
		parameters.output.directory = prm.get("directory");
		parameters.output.basename  = prm.get("basename");
	}
	prm.leave_subsection();

	prm.enter_subsection("Adaptivity");
	{
		parameters.adaptivity.mode                   = prm.get("mode");
		parameters.adaptivity.time_tolerance_high    = prm.get_double("time_tolerance_high");
		parameters.adaptivity.time_tolerance_low     = prm.get_double("time_tolerance_low");
		parameters.adaptivity.refine_every_n_steps   = prm.get_integer("refine_every_n_steps");
		parameters.adaptivity.max_refinement_level   = prm.get_integer("max_refinement_level");
		parameters.adaptivity.space_refine_fraction  = prm.get_double("space_refine_fraction");
		parameters.adaptivity.space_coarsen_fraction = prm.get_double("space_coarsen_fraction");
	}
	prm.leave_subsection();

	validate(parameters);

	return parameters;
}

void write_default_project_parameters(const std::string &file_name)
{
  ParameterHandler prm;
  declare_parameters(prm);

  std::ofstream out(file_name);
  prm.print_parameters(out, ParameterHandler::Text);
}