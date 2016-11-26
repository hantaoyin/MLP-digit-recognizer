// -*- mode:c++; c-basic-offset:2;fill-column:100 -*-
//
// =================================================================================================
// This program is constructed to solve the "Digit Recognizer" tutorial challenge on kaggle.com,
// using a simple MLP.
//
// See https://www.kaggle.com/c/digit-recognizer for more details.
//
// This program is created using the MIT Deep Learning textbook (http://www.deeplearningbook.org/,
// by Ian Goodfellow, Yoshua Bengio, and Aaron Courville) as a reference.
// =================================================================================================
// A simple feedforward network that serves as a classifier has m inputs i[0..(n-1)] and n outputs
// o[0..(n-1)].
//
// For output values:
// 2. Sum{o[k], 0 <= k < n} = 1.
// 3. 0 <= o[k] <= 1 for all k.
//
// For use with digit recognizer:
//
// 1. The input is an array with m elements, where m is the number of pixels in the input data.  For
// this problem, m = 28*28 = 784.
//
// 2. The output is an array with 10 elements.  o[k] is the perceived probability that the image
// represent digit k.
//
// =================================================================================================
// 1. To compile:  clang++ -O3 -Wall mlp-digit_recognizer.C
//
// 2. Input: it expects data/train.csv as training data, data/test.csv as testing data.  Both files
// are downloaded from Kaggle without further modification.
//
// 3. Outut: it output ssubmission.csv which matches the format required by kaggle.com, and can be
// submitted directly.
//
// 4. Other information: debug_draw_greyscale() draws a grayscale image using ASCII characters with
// different pixel densities ("ASCII art").  This is useful for human inspection while debugging.
#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cstdlib>

using namespace std;

namespace mlp {

class Node;

class Edge {
public:
  Edge(size_t size):v(size),dv(size) {}
  double& operator()(size_t id) {
    return v[id];
  }
  const double& operator()(size_t id) const {
    return v[id];
  }
  double& D(size_t id) {
    return dv[id];
  }
  const double& D(size_t id) const {
    return dv[id];
  }
  size_t size(void) const {
    return v.size();
  }

  Node* prev() const {
    return p;
  }
  void set_prev(Node* _p) {
    p = _p;
  }
  Node* next() const {
    return n;
  }
  void set_next(Node* _n) {
    n = _n;
  }
private:
  vector<double> v;
  vector<double> dv;
  Node *p = nullptr;
  Node *n = nullptr;
};

// TODO: Having reference to edges is a bad idea, change it.  Reference may become invalid whenever
// the referred object is moved.
class Node {
public:
  Node(Edge &x, Edge &y):x(x), y(y) {
    x.set_next(this);
    y.set_prev(this);
  }
  virtual void forward(void) = 0;
  virtual void backward_propagate(void) = 0;
  virtual ~Node() {}
protected:
  Edge &x, &y;
};

class ReLU : public Node {
public:
  ReLU(Edge &x, Edge &y):Node(x, y) {
    assert(x.size() == y.size());
  }
  void forward(void) final {
    for (size_t i = 0; i < x.size(); ++i) {
      y(i) = x(i) > 0 ? x(i) : a * x(i);
    }
  }
  void backward_propagate(void) final {
    for (size_t i = 0; i < x.size(); ++i) {
      x.D(i) = y(i) > 0 ? y.D(i) : a * y.D(i);
    }
  }
private:
  const double a = 0.01;
};

class SoftMax : public Node {
public:
  SoftMax(Edge &x, Edge &y):Node(x, y) {
    assert(x.size() == y.size());
  }

  // y[i] = exp(x[i]) / s, where s = Sum{exp(x[i]), {i}}.
  void forward(void) final {
    double xmax = x(0);
    for (size_t i = 1; i < x.size(); ++i) {
      xmax = max(xmax, x(i));
    }

    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
      y(i) = exp(x(i) - xmax);
      sum += y(i);
    }

    assert(sum >= 1.0);
    assert(sum <= 1.01 * x.size());

    sum = 1.0 / sum;
    for (size_t i = 0; i < x.size(); ++i) {
      y(i) *= sum;
    }
  }
  void backward_propagate(void) final {
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
      x.D(i) = y(i) * y.D(i);
      sum += x.D(i);
    }
    for (size_t i = 0; i < x.size(); ++i) {
      x.D(i) -= y(i) * sum;
    }
  }
};

class AffineMap : public Node {
public:
  AffineMap(Edge& x, Edge& y)
    : Node(x, y), w(x.size() * y.size()), b(y.size()) {
    for (auto& wi : w) {
      wi = .1 * ((double)rand() / RAND_MAX - 0.5);
    }
    for (auto& bi : b) {
      bi = .1 * ((double)rand() / RAND_MAX - 0.5);
    }
  }
  void forward(void) final {
    for (size_t i = 0; i < y.size(); ++i) {
      y(i) = b[i];
      for (size_t j = 0; j < x.size(); ++j) {
        y(i) += w[i * x.size() + j] * x(j);
      }
    }
  }
  void backward_propagate(void) final {
    for (size_t j = 0; j < x.size(); ++j) {
      x.D(j) = 0.0;
    }
    for (size_t i = 0; i < y.size(); ++i) {
      for (size_t j = 0; j < x.size(); ++j) {
        x.D(j) += y.D(i) * w[i * x.size() + j];
        w[i * x.size() + j] += y.D(i) * x(j);
      }
      b[i] += y.D(i);
    }
  }
private:
  vector<double> w;
  vector<double> b;
};

// For this type of network, we assume that there is no cross layer link.  In
// other words, the output of node i is always fed into node i + 1.
class SimpleClassifierNetwork {
public:
  SimpleClassifierNetwork(const vector<size_t>& sizes) {
    assert(sizes.size() >= 2);
    const size_t layers = sizes.size() - 1;

    e.emplace_back(sizes[0]);
    for (size_t i = 0; i < layers; ++i) {
      e.emplace_back(sizes[i + 1]);
      e.emplace_back(sizes[i + 1]);
    }

    for (size_t i = 0; i < layers; ++i) {
      const bool hidden_layer = i + 1 < layers;
      v.emplace_back(make_unique<AffineMap>(e[2 * i], e[2 * i + 1]));
      if (hidden_layer) {
        v.emplace_back(make_unique<ReLU>(e[2 * i + 1], e[2 * i + 2]));
      } else {
        v.emplace_back(make_unique<SoftMax>(e[2 * i + 1], e[2 * i + 2]));
      }
    }
  }

  const Edge& forward(const vector<double> &in) {
    assert(in.size() == e[0].size());
    for (size_t i = 0; i < in.size(); ++i) {
      e[0](i) = in[i];
    }
    for (size_t i = 0; i < v.size(); ++i) {
      v[i]->forward();
    }
    return e.back();
  }

  // Training data are (in, out) pairs.  This function only supports training
  // with one example at a time.
  void train_mle(const vector<double>& in, const vector<bool>& out, double step_size) {
    assert(out.size() == e.back().size());

    // cost function is MLE + weight decay.
    forward(in);

    Edge &eb = e.back();
    for (size_t i = 0; i < eb.size(); ++i) {
      eb.D(i) = -step_size * 0.00002 * eb(i);
      if (out[i]) {
        // For SoftMax, unstable.
        eb.D(i) += step_size / eb(i);

        // For LogSoftMax.
        // eb.D(i) += step_size;
      }
    }
    for (size_t i = v.size(); i > 0; --i) {
      v[i - 1]->backward_propagate();
    }

    bool has_nan = false;
    for (size_t k = 0; k < eb.size(); ++k) {
      if (isnan(eb.D(k)) || isnan(eb(k))) has_nan = true;
    }
    static uint64_t iteration = 0;
    if (++iteration == 10 || has_nan) {
      cout.precision(2);
      for (size_t k = 0; k < out.size(); ++k) {
        cout.width(2);
        cout << out[k];
      }
      cout << " ==> ";
      for (size_t k = 0; k < eb.size(); ++k) {
        cout.width(9);
        cout << eb(k);
      }
      cout << " ==> ";
      for (size_t k = 0; k < eb.size(); ++k) {
        cout.width(9);
        cout << eb.D(k);
      }
      cout << " ==> ";
      double mse = 0;
      for (size_t k = 0; k < eb.size(); ++k) {
        double v = (out[k] ? 1 : 0) - eb(k);
        mse += v * v;
      }
      cout.width(10);
      cout << sqrt(mse / 10) << endl;
      iteration = 0;
      // iteration = 0;
      if (has_nan) {
        exit(-1);
      }
    }
  }
private:
  vector<unique_ptr<Node>> v;
  vector<Edge> e;
};
}

struct TrainingData {
  vector<double> image;
  vector<bool> label;
};

void driver(mlp::SimpleClassifierNetwork& scn, const vector<TrainingData>& training_data, uint64_t iterations) {
  for (size_t iteration = 0; iteration < iterations; ++iteration) {
    size_t id = training_data.size() * ((double)rand() / RAND_MAX);
    id %= training_data.size();
    const TrainingData& t = training_data[id];
    const double step_size = iteration < 100000 ? 0.01 : 0.001;
    scn.train_mle(t.image, t.label, step_size);
  }
}

// Draw a simple grey scale image for debugging.
void debug_draw_greyscale(const vector<double>&data, size_t width, size_t height, ofstream& ofs) {
  string colormap(" .:-=+*#%@");
  assert(data.size() == width * height);

  const size_t num_scales = colormap.size() - 1;
  for (size_t i = 1; i <= width * height; ++i) {
    ofs << colormap[size_t(data[i] * num_scales)];
    if (i % width == 0) {
      ofs << "\n";
    }
  }
}

int main() {
  srand(time(0));
  const size_t width = 28;
  const size_t height = 28;

  mlp::SimpleClassifierNetwork scn({width * height, width * height, 100, 100, 30, 10});

  ifstream training_file("data/train.csv");
  vector<TrainingData> training_data;

  // Skip the 1st line, which gives names to each column.
  while (training_file.get() != '\n');

  int label;
  while (true) {
    training_file >> label;
    if (!training_file) break;

    training_data.emplace_back();
    TrainingData& e = training_data.back();
    e.label.assign(10, false);
    e.label[label] = true;

    e.image.resize(width * height);
    for (size_t i = 0; i < width * height; ++i) {
      int pixel;
      training_file.get();
      training_file >> pixel;
      e.image[i] = (double)pixel / 255.0;
    }
    training_file.get();
  }
  driver(scn, training_data, 30000);

  ifstream test_file("data/test.csv");
  ofstream output("submission.csv");

  // Consume the 1st line in the test data file.
  // Print the header as required.
  output << "ImageId,Label\n";
  while (test_file.get() != '\n');

  int id = 0;
  vector<double> image(width * height);
  while (true) {
    for (size_t i = 0; i < width * height; ++i) {
      int pixel;
      test_file >> pixel;
      test_file.get();
      image[i] = (double)pixel / 255.0;
    }
    if (test_file.eof()) break;

    const mlp::Edge& out = scn.forward(image);
    assert(out.size() == 10);
    int max_id = 0;
    for (int i = 1; i < 10; ++i) {
      max_id = out(i) > out(max_id) ? i : max_id;
    }
    output << ++id << "," << max_id << "\n";
    // debug_draw_greyscale(image, width, height, output);
  }
  return 0;
}
