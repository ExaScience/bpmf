#include "io.h"

#include <array>
#include <iostream>
#include <sstream>
#include <cctype>
#include <algorithm>
#include <array>

#include "error.h"
#include "gzstream.h"

#include <unsupported/Eigen/SparseExtra>

#define EXTENSION_SDM ".sdm" //sparse double matrix (binary file)
#define EXTENSION_SBM ".sbm" //sparse binary matrix (binary file)
#define EXTENSION_MTX ".mtx" //sparse matrix (txt file)
#define EXTENSION_MM  ".mm"  //sparse matrix (txt file)
#define EXTENSION_CSV ".csv" //dense matrix (txt file)
#define EXTENSION_DDM ".ddm" //dense double matrix (binary file)

#define EXTENSION_GZ ".gz" // gzipped file

#define MM_OBJ_MATRIX   "MATRIX"
#define MM_FMT_ARRAY    "ARRAY"
#define MM_FMT_COORD    "COORDINATE"
#define MM_FLD_REAL     "REAL"
#define MM_FLD_PATTERN  "PATTERN"
#define MM_SYM_GENERAL  "GENERAL"

MatrixType ExtensionToMatrixType(const std::string& fname)
{
   std::size_t dotIndex = fname.find_last_of(".");
   THROWERROR_ASSERT_MSG(dotIndex != std::string::npos, "Extension is not specified in " + fname);
   std::string extension = fname.substr(dotIndex);

   bool compressed = false;
   if (extension == EXTENSION_GZ)
   {
      compressed = true;
      std::size_t secondDotIndex = fname.find_last_of(".", dotIndex - 1);
      THROWERROR_ASSERT_MSG(dotIndex != std::string::npos, "Extension is not specified in " + fname);
      extension = fname.substr(secondDotIndex, dotIndex - secondDotIndex);
   }

#if 0
   std::cout << "filename: " << fname << std::endl;
   std::cout << "extension: " << extension << std::endl;
   std::cout << "compressed: " << compressed << std::endl;
#endif

   if (extension == EXTENSION_SDM)
   {
      return { MatrixType::sdm, compressed };
   }
   else if (extension == EXTENSION_SBM)
   {
      return { MatrixType::sbm, compressed };
   }
   else if (extension == EXTENSION_MTX || extension == EXTENSION_MM)
   {
       return { MatrixType::mtx, compressed };
   }
   else if (extension == EXTENSION_CSV)
   {
      return { MatrixType::csv, compressed };
   }
   else if (extension == EXTENSION_DDM)
   {
      return { MatrixType::ddm, compressed };
   }
   else
   {
      THROWERROR("Unknown file type: " + extension + " specified in " + fname);
   }
   return { MatrixType::none, compressed };
}

std::string MatrixTypeToExtension(MatrixType matrixType)
{
   std::string extension;

   switch (matrixType.type)
   {
   case MatrixType::sdm:
      extension = EXTENSION_SDM;
      break;
   case MatrixType::sbm:
      extension = EXTENSION_SBM;
      break;
   case MatrixType::mtx:
      extension = EXTENSION_MTX;
      break;
   case MatrixType::csv:
      extension = EXTENSION_CSV;
      break;
   case MatrixType::ddm:
      extension = EXTENSION_DDM;
      break;
   case MatrixType::none:
   default:
      THROWERROR("Unknown matrix type");
   }
   return std::string();
}

void read_matrix(const std::string& filename, Eigen::VectorXd& V)
{
   Eigen::MatrixXd X;
   read_matrix(filename, X);
   V = X; // this will fail if X has more than one column
}

std::istream *open_inputfile(const std::string& filename)
{
   MatrixType matrixType = ExtensionToMatrixType(filename);
   THROWERROR_FILE_NOT_EXIST(filename)

   std::istream *stream;
   if (matrixType.compressed)
   {
      igzstream *fileStream = new igzstream;
      fileStream->open(filename.c_str());
      THROWERROR_ASSERT_MSG(fileStream->good(), "Error opening file: " + filename);
      stream = fileStream;
   } 
   else 
   {
      std::ifstream *fileStream = new std::ifstream(filename, std::ios_base::binary);
      THROWERROR_ASSERT_MSG(fileStream->is_open(), "Error opening file: " + filename);
      stream = fileStream;
   }

   return stream;
}

void read_matrix(const std::string& filename, Eigen::MatrixXd& X)
{
   MatrixType matrixType = ExtensionToMatrixType(filename);
   std::istream *stream = open_inputfile(filename);

   switch (matrixType.type)
   {
   case MatrixType::sdm:
      THROWERROR("Invalid matrix type");
   case MatrixType::sbm:
      THROWERROR("Invalid matrix type");
   case MatrixType::mtx:
      read_matrix_market(*stream, X);
      break;
   case MatrixType::csv:
      read_dense_float64_csv(*stream, X);
      break;
   case MatrixType::ddm:
      read_dense_float64_bin(*stream, X);
      break;
   case MatrixType::none:
      THROWERROR("Unknown matrix type");
   default:
      THROWERROR("Unknown matrix type");
   }

   delete stream;
}

void read_matrix(const std::string& filename, Eigen::SparseMatrix<double>& X)
{
   MatrixType matrixType = ExtensionToMatrixType(filename);
   std::istream *stream = open_inputfile(filename);

   switch (matrixType.type)
   {
   case MatrixType::sdm:
      read_sparse_float64_bin(*stream, X);
      break;
   case MatrixType::sbm:
      read_sparse_binary_bin(*stream, X);
      break;
   case MatrixType::mtx:
      read_matrix_market(*stream, X);
      break;
   case MatrixType::csv:
      THROWERROR("Invalid matrix type");
   case MatrixType::ddm:
      THROWERROR("Invalid matrix type");
   case MatrixType::none:
      THROWERROR("Unknown matrix type");
   default:
      THROWERROR("Unknown matrix type");
   }

   delete stream;
}

void read_dense_float64_bin(std::istream& in, Eigen::MatrixXd& X)
{
   std::uint64_t nrow = 0;
   std::uint64_t ncol = 0;

   in.read(reinterpret_cast<char*>(&nrow), sizeof(std::uint64_t));
   in.read(reinterpret_cast<char*>(&ncol), sizeof(std::uint64_t));

   X.resize(nrow, ncol);
   in.read(reinterpret_cast<char*>(X.data()), nrow * ncol * sizeof(typename Eigen::MatrixXd::Scalar));
}

void read_dense_float64_csv(std::istream& in, Eigen::MatrixXd& X)
{
   std::stringstream ss;
   std::string line;

   // rows and cols
   getline(in, line);
   ss.clear();
   ss << line;
   std::uint64_t nrow;
   ss >> nrow;

   getline(in, line);
   ss.clear();
   ss << line;
   std::uint64_t ncol;
   ss >> ncol;

   X.resize(nrow, ncol);

   std::uint64_t row = 0;
   std::uint64_t col = 0;

   while(getline(in, line) && row < nrow)
   {
      col = 0;

      std::stringstream lineStream(line);
      std::string cell;

      while (std::getline(lineStream, cell, ',') && col < ncol)
      {
         X(row, col++) = stod(cell);
      }

      row++;
   }

   if(row != nrow)
   {
      THROWERROR("invalid number of rows");
   }

   if(col != ncol)
   {
      THROWERROR("invalid number of columns");
   }
}

void read_sparse_float64_bin(std::istream& in, Eigen::SparseMatrix<double>& X)
{
   std::uint64_t nrow;
   std::uint64_t ncol;
   std::uint64_t nnz;

   in.read(reinterpret_cast<char*>(&nrow), sizeof(std::uint64_t));
   in.read(reinterpret_cast<char*>(&ncol), sizeof(std::uint64_t));
   in.read(reinterpret_cast<char*>(&nnz), sizeof(std::uint64_t));

   std::vector<std::uint32_t> rows(nnz);
   in.read(reinterpret_cast<char*>(rows.data()), rows.size() * sizeof(std::uint32_t));
   std::for_each(rows.begin(), rows.end(), [](std::uint32_t& row){ row--; });

   std::vector<std::uint32_t> cols(nnz);
   in.read(reinterpret_cast<char*>(cols.data()), cols.size() * sizeof(std::uint32_t));
   std::for_each(cols.begin(), cols.end(), [](std::uint32_t& col){ col--; });

   std::vector<double> values(nnz);
   in.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(double));

   std::vector<Eigen::Triplet<double> > triplets;
   for(uint64_t i = 0; i < nnz; i++)
      triplets.push_back(Eigen::Triplet<double>(rows[i], cols[i], values[i]));

   X.resize(nrow, ncol);
   X.setFromTriplets(triplets.begin(), triplets.end());

   if(X.nonZeros() != (int)nnz)
   {
      THROWERROR("Invalid number of values");
   }
}

void read_sparse_binary_bin(std::istream& in, Eigen::SparseMatrix<double>& X)
{
   std::uint64_t nrow;
   std::uint64_t ncol;
   std::uint64_t nnz;

   in.read(reinterpret_cast<char*>(&nrow), sizeof(std::uint64_t));
   in.read(reinterpret_cast<char*>(&ncol), sizeof(std::uint64_t));
   in.read(reinterpret_cast<char*>(&nnz), sizeof(std::uint64_t));

   std::vector<std::uint32_t> rows(nnz);
   in.read(reinterpret_cast<char*>(rows.data()), rows.size() * sizeof(std::uint32_t));
   std::for_each(rows.begin(), rows.end(), [](std::uint32_t& row){ row--; });

   std::vector<std::uint32_t> cols(nnz);
   in.read(reinterpret_cast<char*>(cols.data()), cols.size() * sizeof(std::uint32_t));
   std::for_each(cols.begin(), cols.end(), [](std::uint32_t& col){ col--; });

   std::vector<Eigen::Triplet<double> > triplets;
   for(uint64_t i = 0; i < nnz; i++)
      triplets.push_back(Eigen::Triplet<double>(rows[i], cols[i], 1));

   X.resize(nrow, ncol);
   X.setFromTriplets(triplets.begin(), triplets.end());
}

// MatrixMarket format specification
// https://github.com/ExaScience/smurff/files/1398286/MMformat.pdf
void read_matrix_market(std::istream& in, Eigen::MatrixXd& X)
{
   // Check that stream has MatrixMarket format data
   std::array<char, 15> matrixMarketArr;
   in.read(matrixMarketArr.data(), 14);
   std::string matrixMarketStr(matrixMarketArr.begin(), matrixMarketArr.end());
   if (matrixMarketStr != "%%MatrixMarket" && !std::isblank(in.get()))
   {
      std::stringstream ss;
      ss << "Cannot read MatrixMarket from input stream: ";
      ss << "the first 15 characters must be '%%MatrixMarket' followed by at least one blank";
      THROWERROR(ss.str());
   }

   // Parse MatrixMarket header
   std::string headerStr;
   std::getline(in, headerStr);
   std::stringstream headerStream(headerStr);

   std::string object;
   headerStream >> object;
   std::transform(object.begin(), object.end(), object.begin(), ::toupper);

   std::string format;
   headerStream >> format;
   std::transform(format.begin(), format.end(), format.begin(), ::toupper);

   std::string field;
   headerStream >> field;
   std::transform(field.begin(), field.end(), field.begin(), ::toupper);

   std::string symmetry;
   headerStream >> symmetry;
   std::transform(symmetry.begin(), symmetry.end(), symmetry.begin(), ::toupper);

   // Check object type
   if (object != MM_OBJ_MATRIX)
   {
      std::stringstream ss;
      ss << "Invalid MartrixMarket object type: expected 'matrix' but got '" << object << "'";
      THROWERROR(ss.str());
   }

   // Check field type
   if (field != MM_FLD_REAL)
   {
      THROWERROR("Invalid MatrixMarket field type: only 'real' field type is supported");
   }

   // Check symmetry type
   if (symmetry != MM_SYM_GENERAL)
   {
      THROWERROR("Invalid MatrixMarket symmetry type: only 'general' symmetry type is supported");
   }

   // Skip comments and empty lines
   while (in.peek() == '%' || in.peek() == '\n')
      in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

   if (format != MM_FMT_ARRAY)
   {
      THROWERROR("Cannot read a sparse matrix as Eigen::MatrixXd");
   }

   std::uint64_t nrows;
   std::uint64_t ncols;
   in >> nrows >> ncols;

   if (in.fail())
   {
      THROWERROR("Could not get 'rows', 'cols' values for array matrix format");
   }

   X.resize(nrows, ncols);

   for (std::uint64_t col = 0; col < ncols; col++)
   {
      for (std::uint64_t row = 0; row < nrows; row++)
      {
         while (in.peek() == '%' || in.peek() == '\n')
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

         double val;
         in >> val;
         if (in.fail())
         {
            THROWERROR("Could not parse an entry line for array matrix format");
         }

         X(row, col) = val;
      }
   }
}

// MatrixMarket format specification
// https://github.com/ExaScience/smurff/files/1398286/MMformat.pdf
void read_matrix_market(std::istream& in, Eigen::SparseMatrix<double>& X)
{
   // Check that stream has MatrixMarket format data
   std::array<char, 15> matrixMarketArr;
   in.read(matrixMarketArr.data(), 14);
   std::string matrixMarketStr(matrixMarketArr.begin(), matrixMarketArr.end());
   if (matrixMarketStr != "%%MatrixMarket" && !std::isblank(in.get()))
   {
      std::stringstream ss;
      ss << "Cannot read MatrixMarket from input stream: ";
      ss << "the first 15 characters must be '%%MatrixMarket' followed by at least one blank\n";
      ss << "Got: " << matrixMarketStr;
      THROWERROR(ss.str());
   }

   // Parse MatrixMarket header
   std::string headerStr;
   std::getline(in, headerStr);
   std::stringstream headerStream(headerStr);

   std::string object;
   headerStream >> object;
   std::transform(object.begin(), object.end(), object.begin(), ::toupper);

   std::string format;
   headerStream >> format;
   std::transform(format.begin(), format.end(), format.begin(), ::toupper);

   std::string field;
   headerStream >> field;
   std::transform(field.begin(), field.end(), field.begin(), ::toupper);

   std::string symmetry;
   headerStream >> symmetry;
   std::transform(symmetry.begin(), symmetry.end(), symmetry.begin(), ::toupper);

   // Check object type
   if (object != MM_OBJ_MATRIX)
   {
      std::stringstream ss;
      ss << "Invalid MartrixMarket object type: expected 'matrix' but got '" << object << "'";
      THROWERROR(ss.str());
   }

   // Check field type
   if (field != MM_FLD_REAL && field != MM_FLD_PATTERN)
   {
      THROWERROR("Invalid MatrixMarket field type: only 'real' and 'pattern' field types are supported");
   }

   // Check symmetry type
   if (symmetry != MM_SYM_GENERAL)
   {
      THROWERROR("Invalid MatrixMarket symmetry type: only 'general' symmetry type is supported");
   }

   // Skip comments and empty lines
   while (in.peek() == '%' || in.peek() == '\n')
      in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

   if (format != MM_FMT_COORD)
   {
      THROWERROR("Cannot read a dense matrix as Eigen::SparseMatrix<double>");
   }

   std::uint64_t nrows;
   std::uint64_t ncols;
   std::uint64_t nnz;
   in >> nrows >> ncols >> nnz;

   if (in.fail())
   {
      THROWERROR("Could not get 'rows', 'cols', 'nnz' values for coordinate matrix format");
   }

   X.resize(nrows, ncols);

   std::vector<Eigen::Triplet<double> > triplets;
   triplets.reserve(nnz);

   for (std::uint64_t i = 0; i < nnz; i++)
   {
      while (in.peek() == '%' || in.peek() == '\n')
         in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

      std::uint32_t row;
      std::uint32_t col;
      double val;

      if (field == MM_FLD_REAL)
      {
         in >> row >> col >> val;
      }
      else if (field == MM_FLD_PATTERN)
      {
         in >> row >> col;
         val = 1.0;
      }

      if (in.fail())
      {
         THROWERROR("Could not parse an entry line for coordinate matrix format");
      }

      triplets.push_back(Eigen::Triplet<double>(row - 1, col - 1, val));
   }

   X.setFromTriplets(triplets.begin(), triplets.end());
}

// ======================================================================================================


std::ostream *open_outputfile(const std::string& filename)
{
   MatrixType matrixType = ExtensionToMatrixType(filename);
   std::ostream *stream;
   if (matrixType.compressed)
   {
      ogzstream *fileStream = new ogzstream;
      fileStream->open(filename.c_str());
      THROWERROR_ASSERT_MSG(fileStream->good(), "Error opening file: " + filename);
      stream = fileStream;
   } 
   else 
   {
      std::ofstream *fileStream = new std::ofstream(filename, std::ios_base::binary);
      THROWERROR_ASSERT_MSG(fileStream->is_open(), "Error opening file: " + filename);
      stream = fileStream;
   }

   return stream;
}


void write_matrix(const std::string& filename, const Eigen::MatrixXd& X)
{
   MatrixType matrixType = ExtensionToMatrixType(filename);
   std::ostream *stream = open_outputfile(filename);

   switch (matrixType.type)
   {
   case MatrixType::sdm:
      THROWERROR("Invalid matrix type");
   case MatrixType::sbm:
      THROWERROR("Invalid matrix type");
   case MatrixType::mtx:
      write_matrix_market(*stream, X);
      break;
   case MatrixType::csv:
      write_dense_float64_csv(*stream, X);
      break;
   case MatrixType::ddm:
      write_dense_float64_bin(*stream, X);
      break;
   case MatrixType::none:
      THROWERROR("Unknown matrix type");
   default:
      THROWERROR("Unknown matrix type");
   }

   delete stream;
}

void write_matrix(const std::string& filename, const Eigen::SparseMatrix<double>& X)
{
   MatrixType matrixType = ExtensionToMatrixType(filename);
   std::ostream *stream = open_outputfile(filename);

   switch (matrixType.type)
   {
   case MatrixType::sdm:
      write_sparse_float64_bin(*stream, X);
      break;
   case MatrixType::sbm:
      write_sparse_binary_bin(*stream, X);
      break;
   case MatrixType::mtx:
      write_matrix_market(*stream, X);
      break;
   case MatrixType::csv:
      THROWERROR("Invalid matrix type");
   case MatrixType::ddm:
      THROWERROR("Invalid matrix type");
   case MatrixType::none:
      THROWERROR("Unknown matrix type");
   default:
      THROWERROR("Unknown matrix type");
   }

   delete stream;
}

void write_dense_float64_bin(std::ostream& out, const Eigen::MatrixXd& X)
{
  std::uint64_t nrow = X.rows();
  std::uint64_t ncol = X.cols();

  out.write(reinterpret_cast<const char*>(&nrow), sizeof(std::uint64_t));
  out.write(reinterpret_cast<const char*>(&ncol), sizeof(std::uint64_t));
  out.write(reinterpret_cast<const char*>(X.data()), nrow * ncol * sizeof(typename Eigen::MatrixXd::Scalar));
}

const static Eigen::IOFormat csvFormat(6, Eigen::DontAlignCols, ",", "\n");

void write_dense_float64_csv(std::ostream& out, const Eigen::MatrixXd& X)
{
   out << X.rows() << std::endl;
   out << X.cols() << std::endl;
   out << X.format(csvFormat) << std::endl;
}

void write_sparse_float64_bin(std::ostream& out, const Eigen::SparseMatrix<double>& X)
{
   std::uint64_t nrow = X.rows();
   std::uint64_t ncol = X.cols();

   std::vector<uint32_t> rows;
   std::vector<uint32_t> cols;
   std::vector<double> values;

   for (int k = 0; k < X.outerSize(); ++k)
   {
      for (Eigen::SparseMatrix<double>::InnerIterator it(X,k); it; ++it)
      {
         rows.push_back(it.row() + 1);
         cols.push_back(it.col() + 1);
         values.push_back(it.value());
      }
   }

   std::uint64_t nnz = values.size();

   out.write(reinterpret_cast<const char*>(&nrow), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&ncol), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&nnz), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(rows.data()), rows.size() * sizeof(std::uint32_t));
   out.write(reinterpret_cast<const char*>(cols.data()), cols.size() * sizeof(std::uint32_t));
   out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
}

void write_sparse_binary_bin(std::ostream& out, const Eigen::SparseMatrix<double>& X)
{
   std::uint64_t nrow = X.rows();
   std::uint64_t ncol = X.cols();

   std::vector<uint32_t> rows;
   std::vector<uint32_t> cols;

   for (int k = 0; k < X.outerSize(); ++k)
   {
      for (Eigen::SparseMatrix<double>::InnerIterator it(X,k); it; ++it)
      {
         if(it.value() > 0)
         {
            rows.push_back(it.row() + 1);
            cols.push_back(it.col() + 1);
         }
      }
   }

   std::uint64_t nnz = rows.size();

   out.write(reinterpret_cast<const char*>(&nrow), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&ncol), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&nnz), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(rows.data()), rows.size() * sizeof(std::uint32_t));
   out.write(reinterpret_cast<const char*>(cols.data()), cols.size() * sizeof(std::uint32_t));
}

// MatrixMarket format specification
// https://github.com/ExaScience/smurff/files/1398286/MMformat.pdf
void write_matrix_market(std::ostream& out, const Eigen::MatrixXd& X)
{
   out << "%%MatrixMarket ";
   out << MM_OBJ_MATRIX << " ";
   out << MM_FMT_ARRAY << " ";
   out << MM_FLD_REAL << " ";
   out << MM_SYM_GENERAL << std::endl;

   out << X.rows() << " ";
   out << X.cols() << std::endl;

   for (Eigen::SparseMatrix<double>::Index col = 0; col < X.cols(); col++)
      for (Eigen::SparseMatrix<double>::Index row = 0; row < X.rows(); row++)
         out << X(row, col) << std::endl;
}

// MatrixMarket format specification
// https://github.com/ExaScience/smurff/files/1398286/MMformat.pdf
void write_matrix_market(std::ostream& out, const Eigen::SparseMatrix<double>& X)
{
   out << "%%MatrixMarket ";
   out << MM_OBJ_MATRIX << " ";
   out << MM_FMT_COORD << " ";
   out << MM_FLD_REAL << " ";
   out << MM_SYM_GENERAL << std::endl;

   out << X.rows() << " ";
   out << X.cols() << " ";
   out << X.nonZeros() << std::endl;

   for (Eigen::Index i = 0; i < X.outerSize(); ++i)
      for (Eigen::SparseMatrix<double>::InnerIterator it(X, i); it; ++it)
         out << it.row() + 1 << " " << it.col() + 1 << " " << it.value() << std::endl;
}
