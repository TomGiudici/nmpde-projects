#ifndef PRM_PARSE_HPP
#define PRM_PARSE_HPP

#include <string>

/**
	* Strongly typed view of config/default.prm.
	*
	* deal.II's ParameterHandler reads strings from the .prm file. This struct is
	* the typed version used by the rest of the code, which avoids scattering
	* string-based parameter lookups through the solver.
	*/
struct Parameters
{
	// Mesh and finite-element space settings.
	struct Mesh
	{
		std::string  file_name = "..mesh/mesh-cube-10.msh";
		unsigned int degree    = 1;
	};

	// Time-integration settings. min/max_delta_t are reserved for adaptivity.
	struct Time
	{
		double final_time      = 1.0;
		double theta           = 1.0;
		double initial_delta_t = 0.0025;
		double min_delta_t     = 1.0e-6;
		double max_delta_t     = 1.0e-2;
	};

	// Diffusion coefficients
	struct Diffusion
	{
		double mu = 1.0;
	};

	// Parameters of source term
	struct Source
	{
		double       a        = 5.0;
		unsigned int N        = 5;
		double       sigma    = 0.1;
		double       center_x = 0.5;
		double       center_y = 0.5;
		double       center_z = 0.5;
	};

	// Adaptivity controls. The baseline currently parses these values so later
  	// implementation can enable time/space adaptivity without changing main.cpp.
	struct Adaptivity
	{
		std::string  mode                   = "none";
		double       time_tolerance_high    = 1.0e-4;
		double       time_tolerance_low     = 2.0e-6;
		unsigned int refine_every_n_steps   = 5;
		unsigned int max_refinement_level   = 8;
		double       space_refine_fraction  = 0.3;
		double       space_coarsen_fraction = 0.1;
	};

	struct Output
	{
		std::string directory = "output";
		std::string basename  = "heat";
	};

	Mesh       mesh;
	Time       time;
	Diffusion  diffusion;
	Source     source;
	Adaptivity adaptivity;
	Output     output;
};

// Read a .prm file, convert entries to Parameters, and validate them.
Parameters read_project_parameters(const std::string &file_name);

// Utility for regenerating a default .prm file from the declarations.
void write_default_project_parameters(const std::string &file_name);

#endif