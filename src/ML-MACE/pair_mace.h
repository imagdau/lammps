/*
https://github.com/mir-group/pair_allegro
using pair_allegro.h as a template
*/

#ifdef PAIR_CLASS

PairStyle(mace,PairMace)

#else

#ifndef LMP_PAIR_MACE_H
#define LMP_PAIR_MACE_H

#include "pair.h"

#include <torch/torch.h>
#include <vector>

namespace LAMMPS_NS {

class PairMace : public Pair {
 public:
  PairMace(class LAMMPS *);
  virtual ~PairMace();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  virtual double init_one(int, int);
  virtual void init_style();
  void allocate();

  double cutoff;
  torch::jit::Module model;
  torch::Device device = torch::kCPU;
  std::vector<int> type_mapper;

  int batch_size = -1;

 protected:
  int debug_mode = 0;

};

}

#endif
#endif
