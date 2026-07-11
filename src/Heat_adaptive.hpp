#ifndef HEAT_HPP
#define HEAT_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>

using namespace dealii;

class Heat
{
public:
  static constexpr unsigned int dim = 3;
  using Point = dealii::Point<dim>;

  class FunctionU0 : public Function<dim>
  {
  public:
    FunctionU0()
      : Function<dim>()
    {}

    virtual double
    value(const Point & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  Heat(const std::string &mesh_file_name_,
       const unsigned int &r_,
       const double &T_,
       const double &theta_,
       const double &delta_t_,
       const std::function<double(const Point &)> &mu_,
       const std::function<double(const Point &, const double &)> &f_);

  void
  run();

protected:
  void
  setup();

  void
  setup_system();

  void
  assemble();

  void
  solve_linear_system();

  void
  refine_mesh();

  void
  output() const;

  const std::string mesh_file_name;
  const unsigned int r;
  const double T;
  const double theta;
  const double delta_t;

  double time = 0.0;
  unsigned int timestep_number = 0;

  std::function<double(const Point &)> mu;
  std::function<double(const Point &, const double &)> f;

  const unsigned int mpi_size;
  const unsigned int mpi_rank;

  parallel::distributed::Triangulation<dim> mesh;
  std::unique_ptr<FiniteElement<dim>> fe;
  std::unique_ptr<Quadrature<dim>> quadrature;
  DoFHandler<dim> dof_handler;

  TrilinosWrappers::SparseMatrix system_matrix;
  TrilinosWrappers::MPI::Vector system_rhs;
  TrilinosWrappers::MPI::Vector solution_owned;
  TrilinosWrappers::MPI::Vector solution;

  ConditionalOStream pcout;

  const unsigned int refine_every_n_steps = 5;
  const double refine_fraction = 0.30;
  const double coarsen_fraction = 0.03;
  const unsigned int max_global_levels = 6;
};

#endif