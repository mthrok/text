#ifdef ENABLE_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex.h>
#include <regex_tokenizer.h>
#include <sentencepiece.h>
#include <vectors.h>
#include <vocab.h>
#endif // ENABLE_PYBIND11

namespace torchtext {

#ifdef ENABLE_PYBIND11

namespace py = pybind11;
// Registers our custom classes with pybind11.
PYBIND11_MODULE(_torchtext, m) {
  // Classes
  py::class_<Regex>(m, "Regex")
      .def(py::init<std::string>())
      .def("Sub", &Regex::Sub);

  py::class_<RegexTokenizer>(m, "RegexTokenizer")
      .def_readonly("patterns_", &RegexTokenizer::patterns_)
      .def_readonly("replacements_", &RegexTokenizer::replacements_)
      .def_readonly("to_lower_", &RegexTokenizer::to_lower_)
      .def(py::init<std::vector<std::string>, std::vector<std::string>, bool>())
      .def("forward", &RegexTokenizer::forward);

  py::class_<SentencePiece>(m, "SentencePiece")
      .def("Encode", &SentencePiece::Encode)
      .def("EncodeAsIds", &SentencePiece::EncodeAsIds)
      .def("EncodeAsPieces", &SentencePiece::EncodeAsPieces)
      .def("GetPieceSize", &SentencePiece::GetPieceSize)
      .def("unk_id", &SentencePiece::unk_id)
      .def("PieceToId", &SentencePiece::PieceToId)
      .def("IdToPiece", &SentencePiece::IdToPiece);

  py::class_<Vectors>(m, "Vectors")
      .def(py::init<std::vector<std::string>, std::vector<int64_t>,
                    torch::Tensor, torch::Tensor>())
      .def_readonly("vectors_", &Vectors::vectors_)
      .def_readonly("unk_tensor_", &Vectors::unk_tensor_)
      .def("get_stoi", &Vectors::get_stoi)
      .def("__getitem__", &Vectors::__getitem__)
      .def("lookup_vectors", &Vectors::lookup_vectors)
      .def("__setitem__", &Vectors::__setitem__)
      .def("__len__", &Vectors::__len__);

  py::class_<Vocab>(m, "Vocab")
      .def(py::init<std::vector<std::string>, std::string>())
      .def_readonly("itos_", &Vocab::itos_)
      .def_readonly("unk_token_", &Vocab::unk_token_)
      .def("__getitem__", &Vocab::__getitem__)
      .def("__len__", &Vocab::__len__)
      .def("insert_token", &Vocab::insert_token)
      .def("append_token", &Vocab::append_token)
      .def("lookup_token", &Vocab::lookup_token)
      .def("lookup_tokens", &Vocab::lookup_tokens)
      .def("lookup_indices", &Vocab::lookup_indices)
      .def("get_stoi", &Vocab::get_stoi)
      .def("get_itos", &Vocab::get_itos);

  // Functions
  m.def("_load_token_and_vectors_from_file",
        &_load_token_and_vectors_from_file);
  m.def("_load_vocab_from_file", &_load_vocab_from_file);
}

#endif // ENABLE_PYBIND11

} // namespace torchtext
