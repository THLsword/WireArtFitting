#ifndef _SEG_IO_H_
#define _SEG_IO_H_ 1

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

//
// Class SegIO
// Format:
//   num_faces
//   [segment_label]*

//
class SegIO {
public:
	std::vector<unsigned int> Data; // Data[face_id] = segment_id 

	~SegIO() {
		Data.clear();
	}
};

inline 
std::ostream& operator << (std::ostream& out, const SegIO& S) {
	for (std::vector<unsigned int>::const_iterator it=S.Data.begin(); it!=S.Data.end(); ++it) {
		out << *it << std::endl;
	}
	return out;
}

inline 
std::istream& operator >> (std::istream& in, SegIO& S) {
	while (in.good()) {
		std::string line;
		std::getline(in, line);
		if (!line.empty()) {
			S.Data.push_back(atoi(line.c_str()));
		}
	}
	return in;
}

inline
void LoadSeg(const std::string& file, SegIO& seg) {
	std::ifstream input(file.c_str());
	input >> seg;
	input.close();
}

inline
void WriteSeg(const std::string& file, const SegIO& seg) {
	std::ofstream output(file.c_str());
	output << seg;
	output.close();
}

#endif
