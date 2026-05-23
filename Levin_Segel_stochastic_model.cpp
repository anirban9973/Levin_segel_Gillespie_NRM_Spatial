
/* Levin-Segel stochastic predator-prey model
   Spatial Gillespie / Next Reaction Method on a 1D periodic lattice.
   Power Spectrum at omega=0 and k=0 — Structure factor calculation.
   OOP + OpenMP version: each thread runs an independent realization.
   All physics and job parameters are read from input.dat at runtime.
*/

#include <cmath>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <limits>
#include <random>
#include <omp.h>
#include <highfive/H5File.hpp>

#include "nrm_heap_header.h"

//----------------------------------------
// Parameter container — populated from input.dat
//----------------------------------------

struct Params {
  long   L;
  double V;
  double b, e, p1, p2, d1P, d2P, d1H, d2H;
  double z, mu1P, mu2P, nu1H, nu2H;
  double t_transient;
  std::vector<double> T_int;   // integration windows, e.g. {1, 32, 4096}
  int    n_threads, realizations_per_thread;
};

Params read_params(const std::string& filename) {
  std::map<std::string, double> raw;
  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Cannot open parameter file: " << filename << "\n";
    std::exit(1);
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    std::string key; double val;
    if (iss >> key >> val) raw[key] = val;
  }

  Params p;
  p.L                      = static_cast<long>(raw.at("L"));
  p.V                      = raw.at("V");
  p.b                      = raw.at("b");
  p.e                      = raw.at("e");
  p.p1                     = raw.at("p1");
  p.p2                     = raw.at("p2");
  p.d1P                    = raw.at("d1P");
  p.d2P                    = raw.at("d2P");
  p.d1H                    = raw.at("d1H");
  p.d2H                    = raw.at("d2H");
  p.z                      = raw.at("z");
  p.mu1P                   = raw.at("mu1P");
  p.mu2P                   = raw.at("mu2P");
  p.nu1H                   = raw.at("nu1H");
  p.nu2H                   = raw.at("nu2H");
  p.T_int = { raw.at("T_int_1"), raw.at("T_int_2"), raw.at("T_int_3") };
  p.t_transient            = raw.at("t_transient");
  p.n_threads              = static_cast<int>(raw.at("n_threads"));
  p.realizations_per_thread = static_cast<int>(raw.at("realizations_per_thread"));
  return p;
}

//----------------------------------------
// Simulation class
//----------------------------------------

class Simulation {

private:

  // Lattice and volume
  long   L;
  double V;

  // Reaction rates
  double b, e, p1, p2, d1P, d2P, d1H, d2H;

  // Diffusion rates
  double z, mu1P, mu2P, nu1H, nu2H;

  // Temporal parameters
  double delta_t, t_transient, t_final;
  std::vector<double> T_int;

  // Seed stored for error reporting
  unsigned int seed;

  // State
  std::vector<long> prey, predator;
  std::vector<std::vector<double>> propensity;
  std::vector<Element> time_heap;
  std::vector<int> SpatialList;

  // Time-integrated spectrum accumulators — one per integration window
  std::vector<std::vector<double>> prey_spectra, pred_spectra;

  // Per-instance RNG — ensures thread safety
  std::mt19937 gen;
  std::uniform_real_distribution<> real_ran;
  std::uniform_int_distribution<> int_ran;

  void compute_propensities(long site) {

    propensity[site].clear();

    double TDiffPredator1 = nu1H * predator[site];
    double TDiffPredator2 = nu2H * predator[site] * (predator[site] - 1) / V;
    double TDiffPrey1     = mu1P * prey[site];
    double TDiffPrey2     = mu2P * prey[site] * predator[site] / V;
    double T1 = prey[site] * b;
    double T2 = prey[site] * (prey[site] - 1) * e / V;
    double T3 = prey[site] * predator[site] * p1 / V;
    double T4 = predator[site] * (predator[site] - 1) * d2H / V;

    propensity[site].push_back(TDiffPredator1); // predator diffusion
    propensity[site].push_back(TDiffPredator2); // predator diffusion coupled
    propensity[site].push_back(TDiffPrey1);     // prey diffusion
    propensity[site].push_back(TDiffPrey2);     // prey diffusion coupled
    propensity[site].push_back(T1);             // birth of prey
    propensity[site].push_back(T2);             // assisted birth of prey
    propensity[site].push_back(T3);             // predation
    propensity[site].push_back(T4);             // death due to competition
  }

public:

  Simulation(const Params& p, unsigned int seed_val) :
    L(p.L), V(p.V),
    b(p.b), e(p.e), p1(p.p1), p2(p.p2),
    d1P(p.d1P), d2P(p.d2P), d1H(p.d1H), d2H(p.d2H),
    z(p.z), mu1P(p.mu1P), mu2P(p.mu2P), nu1H(p.nu1H), nu2H(p.nu2H),
    delta_t(1.0),
    t_transient(p.t_transient),
    t_final(p.t_transient + *std::max_element(p.T_int.begin(), p.T_int.end()) + 5.0),
    T_int(p.T_int),
    seed(seed_val),
    prey(p.L, 0), predator(p.L, 0),
    propensity(p.L),
    time_heap(p.L),
    SpatialList(p.L),
    prey_spectra(p.T_int.size(), std::vector<double>(p.L, 0.0)),
    pred_spectra(p.T_int.size(), std::vector<double>(p.L, 0.0)),
    gen(seed_val),
    real_ran(0.0, 1.0),
    int_ran(0, static_cast<int>(p.L - 1))
  {}

  void initialize() {

    // Mean-field steady-state initial conditions
    double denom   = p1 * p1 - d2H * e;
    double rhoP    = (b * d2H) / denom;
    double rhoH    = (b * p1)  / denom;

    double init_prey_den     = rhoP * V;
    double init_predator_den = rhoH * V;
	mu1P = z * mu1P;
	mu2P = z * mu2P;
	nu1H = z * nu1H;
	nu2H = z * nu2H;

    for (int i = 0; i < init_predator_den * L; i++)
      predator[int_ran(gen)]++;

    for (int i = 0; i < init_prey_den * L; i++)
      prey[int_ran(gen)]++;

    for (long site = 0; site < L; site++) {
      compute_propensities(site);
      double rate = std::accumulate(propensity[site].begin(), propensity[site].end(), 0.0);
      std::exponential_distribution<double> exp_dist(rate);
      time_heap[site].even = exp_dist(gen);
      time_heap[site].odd  = site;
    }

    std::make_heap(time_heap.begin(), time_heap.end(), compareEven);

    for (size_t i = 0; i < time_heap.size(); i++)
      SpatialList[time_heap[i].odd] = i;
  }

  void run() {

    size_t n_windows = T_int.size();
    std::vector<double> data_collection_time(n_windows, t_transient);
    std::vector<long>   collected_data_count(n_windows, 0L);
    std::vector<int> tag_sites;

    while (time_heap[0].even < t_final) {

      int    AffectedNeighbouringSite = L;
      double current_time = time_heap[0].even;

      double rand_num = real_ran(gen);
      long   site = time_heap[0].odd;
      long   ln   = (site - 1 + L) % L;
      long   rn   = (site + 1)     % L;

      double total_propensity = std::accumulate(propensity[site].begin(),
                                                propensity[site].end(), 0.0);
      double cumu_prop = 0.0;
      long   i = 0;

      for (const double& elements : propensity[site]) {
        i++;
        if (rand_num * total_propensity >= cumu_prop &&
            rand_num * total_propensity <  cumu_prop + elements)
          break;
        cumu_prop += elements;
      }

      // execute selected event
      if (i == 1) {
        predator[site]--;
        if (real_ran(gen) < 0.5) { predator[ln]++; AffectedNeighbouringSite = ln; }
        else                     { predator[rn]++; AffectedNeighbouringSite = rn; }
      }
      else if (i == 2) {
        predator[site]--;
        if (real_ran(gen) < 0.5) { predator[ln]++; AffectedNeighbouringSite = ln; }
        else                     { predator[rn]++; AffectedNeighbouringSite = rn; }
      }
      else if (i == 3) {
        prey[site]--;
        if (real_ran(gen) < 0.5) { prey[ln]++; AffectedNeighbouringSite = ln; }
        else                     { prey[rn]++; AffectedNeighbouringSite = rn; }
      }
      else if (i == 4) {
        prey[site]--;
        if (real_ran(gen) < 0.5) { prey[ln]++; AffectedNeighbouringSite = ln; }
        else                     { prey[rn]++; AffectedNeighbouringSite = rn; }
      }
      else if (i == 5) { prey[site]++;                   } // birth of prey
      else if (i == 6) { prey[site]++;                   } // nonlinear birth of prey
      else if (i == 7) { prey[site]--; predator[site]++; } // predation
      else if (i == 8) { predator[site]--;               } // death due to competition

      if (prey[site] < 0 || predator[site] < 0) {
        std::cout << "code error\n"
                  << site << "\t" << prey[site] << "\t" << predator[site] << "\n"
                  << "event = " << i << "\n"
                  << "corresponding seed number = " << seed << "\n";
        std::string errfile = "config_error_" + std::to_string(seed) + ".dat";
        std::ofstream fp(errfile);
        for (long j = 0; j < L; j++)
          fp << j << "\t" << prey[j] << "\t" << predator[j] << "\n";
        fp.close();
        break;
      }

      // time-integrated data collection — independent for each window
      for (size_t j = 0; j < n_windows; j++) {
        long max_count = static_cast<long>(T_int[j] / delta_t);
        if (current_time > data_collection_time[j] && collected_data_count[j] < max_count) {
          for (long r = 0; r < L; r++) {
            prey_spectra[j][r] += prey[r]     * delta_t;
            pred_spectra[j][r] += predator[r] * delta_t;
          }
          data_collection_time[j] += delta_t;
          collected_data_count[j]++;
        }
      }

      // propensity update for event site and (if diffusion) affected neighbor
      tag_sites.clear();
      tag_sites.push_back(static_cast<int>(site));
      if (AffectedNeighbouringSite != L)
        tag_sites.push_back(AffectedNeighbouringSite);

      for (const int& elem : tag_sites) {

        long s = elem;

        double PreviousPropensity = std::accumulate(propensity[s].begin(),
                                                    propensity[s].end(), 0.0);
        compute_propensities(s);
        double NewPropensity = std::accumulate(propensity[s].begin(),
                                               propensity[s].end(), 0.0);

        if (s == tag_sites[0]) {  // event site — always at heap root
          if (NewPropensity != 0.0) {
            std::exponential_distribution<double> exp_dist(NewPropensity);
            time_heap[0].even = current_time + exp_dist(gen);
          } else {
            time_heap[0].even = std::numeric_limits<double>::max();
          }
          heapify(time_heap, 0, SpatialList);
        }
        else if (tag_sites.size() > 1) {  // affected neighbor — NRM time rescaling
          if (PreviousPropensity != 0.0) {
            time_heap[SpatialList[s]].even = current_time +
              (time_heap[SpatialList[s]].even - current_time) *
              (PreviousPropensity / NewPropensity);
            heapify(time_heap, SpatialList[s], SpatialList);
          }
          else if (PreviousPropensity == 0.0 && NewPropensity != 0.0) {
            std::exponential_distribution<double> exp_dist(NewPropensity);
            time_heap[SpatialList[s]].even = current_time + exp_dist(gen);
            heapify(time_heap, SpatialList[s], SpatialList);
          }
        }
      }

    } // end of dynamics
  }

  void finalize() {}  // accumulators already hold raw time integrals

  const std::vector<double>& prey_spectrum(int j)     const { return prey_spectra[j]; }
  const std::vector<double>& predator_spectrum(int j) const { return pred_spectra[j]; }
  unsigned int get_seed() const { return seed; }

};

//----------------------------------------

int main(int argc, char *argv[]) {

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <job_id>" << std::endl;
    return 1;
  }

  int job_id = std::atoi(argv[1]);

  Params params = read_params("input.dat");

  std::cout << "Job " << job_id << " — "
            << params.n_threads << " threads, "
            << params.realizations_per_thread << " realizations/thread, "
            << params.n_threads * params.realizations_per_thread << " total.\n";

  int n_threads              = params.n_threads;
  int realizations_per_thread = params.realizations_per_thread;
  int realizations_per_job   = n_threads * realizations_per_thread;

  size_t R_total = static_cast<size_t>(realizations_per_job);
  size_t L       = static_cast<size_t>(params.L);

  // One HDF5 file per array job, created before the parallel region (thread-safe)
  std::string h5file = "output_job" + std::to_string(job_id) + ".h5";
  HighFive::File file(h5file, HighFive::File::Truncate);

  // Build dataset names from T_int values
  std::vector<std::string> T_labels;
  for (double t : params.T_int)
    T_labels.push_back(std::to_string(static_cast<long>(t)));

  std::vector<HighFive::DataSet> prey_dsets, pred_dsets;
  for (const auto& lbl : T_labels) {
    prey_dsets.push_back(file.createDataSet<double>("prey_T"     + lbl, HighFive::DataSpace({R_total, L})));
    pred_dsets.push_back(file.createDataSet<double>("predator_T" + lbl, HighFive::DataSpace({R_total, L})));
  }
  auto seed_dset = file.createDataSet<unsigned int>("seeds", HighFive::DataSpace({R_total}));

  #pragma omp parallel num_threads(n_threads)
  {
    int thread_id = omp_get_thread_num();

    int n_windows = static_cast<int>(params.T_int.size());

    // Per-window buffers: [window][realization][site]
    std::vector<std::vector<std::vector<double>>> prey_bufs(n_windows,
        std::vector<std::vector<double>>(realizations_per_thread));
    std::vector<std::vector<std::vector<double>>> pred_bufs(n_windows,
        std::vector<std::vector<double>>(realizations_per_thread));
    std::vector<unsigned int> seed_buf(realizations_per_thread);

    for (int local_idx = 0; local_idx < realizations_per_thread; local_idx++) {

      unsigned int seed = static_cast<unsigned int>(
        (job_id - 1) * realizations_per_job +
        thread_id    * realizations_per_thread + local_idx);

      Simulation sim(params, seed);
      sim.initialize();
      sim.run();
      sim.finalize();

      for (int j = 0; j < n_windows; j++) {
        prey_bufs[j][local_idx] = sim.prey_spectrum(j);
        pred_bufs[j][local_idx] = sim.predator_spectrum(j);
      }
      seed_buf[local_idx] = seed;
    }

    // All realizations done — dump the whole block in one critical section.
    // Row range for this thread: [thread_id*R, (thread_id+1)*R)
    size_t R         = static_cast<size_t>(realizations_per_thread);
    size_t row_start = static_cast<size_t>(thread_id) * R;

    #pragma omp critical (hdf5)
    {
      for (int j = 0; j < n_windows; j++) {
        prey_dsets[j].select({row_start, 0}, {R, L}).write(prey_bufs[j]);
        pred_dsets[j].select({row_start, 0}, {R, L}).write(pred_bufs[j]);
      }
      seed_dset.select({row_start}, {R}).write(seed_buf);
      file.flush();
    }

  } // end parallel

  return 0;
}
