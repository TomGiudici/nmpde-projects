#include "Heat_adaptive.hpp"

Heat::Heat(const std::string &mesh_file_name_,
           const unsigned int &r_,
           const double &T_,
           const double &theta_,
           const double &delta_t_,
           const std::function<double(const Point &)> &mu_,
           const std::function<double(const Point &, const double &)> &f_)
  : mesh_file_name(mesh_file_name_)
  , r(r_)
  , T(T_)
  , theta(theta_)
  , delta_t(delta_t_)
  , mu(mu_)
  , f(f_)
  , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
  , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
  , mesh(MPI_COMM_WORLD)
  , dof_handler(mesh)
  , pcout(std::cout, mpi_rank == 0)
{}

void
Heat::setup()
{
  pcout << "===============================================\n";
  pcout << "Initializing the mesh" << std::endl;

  GridGenerator::hyper_cube(mesh, 0.0, 1.0);
  mesh.refine_global(1);  // or 2 if you want a slightly finer initial mesh

  pcout << " Number of elements = " << mesh.n_global_active_cells()
        << std::endl;
  pcout << "-----------------------------------------------\n";

  pcout << "Initializing the finite element space" << std::endl;
  fe = std::make_unique<FE_Q<dim>>(r);
  quadrature = std::make_unique<QGauss<dim>>(r + 1);

  pcout << " Degree = " << fe->degree << std::endl;
  pcout << " DoFs per cell = " << fe->dofs_per_cell << std::endl;
  pcout << " Quadrature points per cell = " << quadrature->size()
        << std::endl;
  pcout << "-----------------------------------------------\n";

  setup_system();
}

void
Heat::setup_system()
{
  pcout << "Initializing the DoF handler" << std::endl;

  dof_handler.distribute_dofs(*fe);

  pcout << " Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  pcout << "-----------------------------------------------\n";

  pcout << "Initializing the linear system" << std::endl;

  const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);

  TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                             locally_owned_dofs,
                                             MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(dof_handler, sparsity);
  sparsity.compress();

  system_matrix.reinit(sparsity);
  system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

  pcout << "-----------------------------------------------\n";
}

void
Heat::assemble()
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_matrix = 0.0;
  system_rhs    = 0.0;

  std::vector<double> solution_old_values(n_q);
  std::vector<Tensor<1, dim>> solution_old_grads(n_q);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);
      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      fe_values.get_function_values(solution, solution_old_values);
      fe_values.get_function_gradients(solution, solution_old_grads);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const Point qp      = fe_values.quadrature_point(q);
          const double mu_loc = mu(qp);

          const double f_old_loc = f(qp, time - delta_t);
          const double f_new_loc = f(qp, time);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) +=
                    (1.0 / delta_t) * fe_values.shape_value(i, q) *
                    fe_values.shape_value(j, q) * fe_values.JxW(q);

                  cell_matrix(i, j) +=
                    theta * mu_loc *
                    scalar_product(fe_values.shape_grad(i, q),
                                   fe_values.shape_grad(j, q)) *
                    fe_values.JxW(q);
                }

              cell_rhs(i) +=
                (1.0 / delta_t) * fe_values.shape_value(i, q) *
                solution_old_values[q] * fe_values.JxW(q);

              cell_rhs(i) -=
                (1.0 - theta) * mu_loc *
                scalar_product(fe_values.shape_grad(i, q),
                               solution_old_grads[q]) *
                fe_values.JxW(q);

              cell_rhs(i) +=
                (theta * f_new_loc + (1.0 - theta) * f_old_loc) *
                fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);
      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}

void
Heat::solve_linear_system()
{
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(
    system_matrix,
    TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  ReductionControl solver_control(10000, 1.0e-16, 1.0e-6);
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);

  pcout << solver_control.last_step() << " CG iterations" << std::endl;
}

void
Heat::refine_mesh()
{
  if (mesh.n_global_levels() >= max_global_levels)
    return;

  Vector<float> estimated_error_per_cell(mesh.n_active_cells());

  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim - 1>(r + 1),
                                     std::map<types::boundary_id,
                                              const Function<dim> *>(),
                                     solution,
                                     estimated_error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_number(mesh,
                                                  estimated_error_per_cell,
                                                  refine_fraction,
                                                  coarsen_fraction);

  for (const auto &cell : mesh.active_cell_iterators())
    if (cell->level() >= max_global_levels - 1)
      cell->clear_refine_flag();

  parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector>
    solution_transfer(dof_handler);

  solution_transfer.prepare_for_coarsening_and_refinement(solution_owned);
  mesh.execute_coarsening_and_refinement();

  dof_handler.clear();
  dof_handler.reinit(mesh);
  setup_system();

  TrilinosWrappers::MPI::Vector transferred_solution;
  transferred_solution.reinit(dof_handler.locally_owned_dofs(),
                              MPI_COMM_WORLD);

  solution_transfer.interpolate(transferred_solution);

  solution_owned = transferred_solution;
  solution       = solution_owned;

  pcout << " Mesh adapted: "
        << mesh.n_global_active_cells()
        << " active cells, "
        << dof_handler.n_dofs()
        << " DoFs" << std::endl;
}

void
Heat::output() const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "solution");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::filesystem::path mesh_path(mesh_file_name);
  const std::string output_file_name = "output-" + mesh_path.stem().string();

  data_out.write_vtu_with_pvtu_record("./",
                                      output_file_name,
                                      timestep_number,
                                      MPI_COMM_WORLD);
}

void
Heat::run()
{
  setup();

  VectorTools::interpolate(dof_handler, FunctionU0(), solution_owned);
  solution = solution_owned;

  time            = 0.0;
  timestep_number = 0;

  output();
  pcout << "===============================================\n";

  while (time < T - 0.5 * delta_t)
    {
      time += delta_t;
      ++timestep_number;

      pcout << "Timestep " << std::setw(3) << timestep_number
            << ", time = " << std::setw(6) << std::fixed
            << std::setprecision(4) << time << " : ";

      assemble();
      solve_linear_system();

      solution = solution_owned;

      if (timestep_number % refine_every_n_steps == 0)
        refine_mesh();

      output();
    }
}