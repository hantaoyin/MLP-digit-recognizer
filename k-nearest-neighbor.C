// -*- mode:c++; c-basic-offset:2;fill-column:100 -*-
//
// =================================================================================================
// This program is constructed to solve the "Digit Recognizer" tutorial challenge on kaggle.com,
// using the k-nearest neighbor algorithm.
//
// See https://www.kaggle.com/c/digit-recognizer for more details.
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
#include <random>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cstdlib>

#ifdef NDEBUG
#   define ASSERT(condition, message) do { } while (false)
#else
#   define ASSERT(condition, message)                                   \
  do {                                                                  \
    if (! (condition)) {                                                \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__  \
                << " line " << __LINE__ << ": " << message << std::endl;\
      std::terminate();                                                 \
    }                                                                   \
  } while (false)
#endif

const size_t W = 28;
const size_t H = 28;

using Image = std::array<double, W * H>;

struct TrainingData {
  Image image;
  size_t label;
};

double distance_l2(const Image& a, const Image &b) {
  double ret = 0;
  for (size_t i = 0; i < W * H; ++i) {
    ret += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return ret;
}

// Algorithm: find the K nearest neighbors and count the occurrence of each label in these
// neighbors.  The label with largest count is used to label the input.
class KNearestNeighbor {
public:
  KNearestNeighbor() {
    std::random_device rd;
    engine.seed(rd());
  }

  void add_training_data(const TrainingData &data) {
    assert(data.label < 10);
    training_data[data.label].push_back(data.image);
  }

  size_t label_v1(const Image &data) {
    std::map<double, size_t> m;
    for (size_t i = 0; i < 10; ++i) {
      for (const auto &image : training_data[i]) {
        const double d = distance_l2(data, image);
        if (m.size() < K || d < m.rbegin()->first) {
          m.emplace(d, i);
        }
        if (m.size() > K) {
          auto iter = m.end();
          m.erase(--iter);
        }
      }
    }
    std::array<size_t, 10> count{};
    for (const auto &kv : m) {
      ++count[kv.second];
    }
    size_t max_count = 0;
    size_t max = 10;
    for (size_t i = 0; i < 10; ++i) {
      // std::cout << std::setw(4) << count[i] << " ";
      if (count[i] > max_count) {
        max_count = count[i];
        max = i;
      }
    }
    // std::cout << "\n";
    assert(max < 10);
    return max;
  }
private:
  const size_t K = 1;
  std::array<std::vector<Image>, 10> training_data;
  std::uniform_real_distribution<double> dist;
  std::default_random_engine engine;
};

// Draw a simple grey scale image for debugging.
static const char colormap[] = " .:-=+*#%@";
void debug_draw_greyscale(const Image&data, std::ofstream& ofs) {
  for (size_t i = 0; i < W * H; ++i) {
    ofs << colormap[static_cast<size_t>(data[i] * (sizeof(colormap) - 2))];
    if ((i + 1) % W == 0) {
      ofs << "\n";
    }
  }
}

void rand_translate_image(const Image& orig, Image& translated) {
  const size_t dw = W * static_cast<double>(rand()) / RAND_MAX;
  const size_t dh = H * static_cast<double>(rand()) / RAND_MAX;
  for (size_t w = 0; w < W; ++w) {
    for (size_t h = 0; h < H; ++h) {
      translated[h * W + w] = orig[(h + dh) % H * W + (w + dw) % W];
    }
  }
}

int main() {
  srand(time(0));

  std::ifstream training_file("data/train.csv");
  std::vector<TrainingData> training_data;

  // Skip the 1st line, which gives names to each column.
  while (training_file.get() != '\n');

  KNearestNeighbor knn;

  size_t label;
  while (true) {
    training_file >> label;
    if (!training_file) break;

    TrainingData e;
    e.label = label;

    for (size_t i = 0; i < W * H; ++i) {
      int pixel;
      training_file.get();
      training_file >> pixel;
      e.image[i] = pixel / 255.;
    }
    training_file.get();
    knn.add_training_data(e);
  }

  std::ifstream test_file("data/test.csv");
  std::ofstream output("submission.csv");

  // Consume the 1st line in the test data file.
  // Print the header as required.
  output << "ImageId,Label\n";
  while (test_file.get() != '\n');

  int id = 0;
  Image image;
  while (true) {
    for (size_t i = 0; i < W * H; ++i) {
      int pixel;
      test_file >> pixel;
      test_file.get();
      image[i] = pixel / 255.;
    }
    if (test_file.eof()) break;

    output << ++id << "," << knn.label_v1(image) << "\n";
    debug_draw_greyscale(image, output);
  }
  return 0;
}
