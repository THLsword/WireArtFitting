#ifndef _ATLAS_IO_H_
#define _ATLAS_IO_H_ 1

#include <map>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

//
// Formats:
// * AtlasIO:
//    num_faces  [chart_label]*
//    num_charts [num_vertices [vertex_id u v]*]*
//    num_charts [r g b]*
//

class AtlasIO {
public:
	enum { NoChart = -1 };
    typedef std::pair<float, float> Vec2;
	typedef std::map<unsigned int, Vec2> ChartIO;

	std::vector<int>     Labels;
	std::vector<ChartIO> Charts;
	std::vector<float>   Colors;

	AtlasIO() {
	}
    
	~AtlasIO() {
		Labels.clear();
		Charts.clear();
		Colors.clear();
	}
		
	unsigned int numCharts() const {
		int N = NoChart;
		for (unsigned int i=0; i<Labels.size(); ++i) {
			N = std::max(N, Labels[i]);
		}
		return static_cast<unsigned int>(N + 1);
	}
};

inline 
std::ostream& operator << (std::ostream& out, const AtlasIO& atlas) {
	out << atlas.Labels.size() << std::endl;
	for (std::vector<int>::const_iterator it=atlas.Labels.begin(); it!=atlas.Labels.end(); ++it) {
		out << *it << std::endl;
	}
	
	out << atlas.Charts.size() << std::endl;
	for (std::vector<AtlasIO::ChartIO>::const_iterator it=atlas.Charts.begin(); it!=atlas.Charts.end(); ++it) {
		out << it->size() << std::endl;
		for (AtlasIO::ChartIO::const_iterator it2=it->begin(); it2!=it->end(); ++it2) {
			out << it2->first << " " << it2->second.first << " " << it2->second.second << std::endl;
		}
	}
	
	out << atlas.Colors.size() / 3 << std::endl;
	for (unsigned int i=0; i<atlas.Colors.size(); i+=3) {
        for (unsigned int j=0; j<3; ++j) {
            out << atlas.Colors[i+j] << " ";
        }
        out << std::endl;
	}
	
	return out;
}

inline 
std::istream& operator >> (std::istream& in, AtlasIO& atlas) {
	unsigned int N;
	
	in >> N;
	atlas.Labels.resize(N, AtlasIO::NoChart);
	for (unsigned int i=0; i<N; ++i) {
		in >> atlas.Labels[i];
	}
	
	in >> N;
	atlas.Charts.resize(N);
	for (unsigned int i=0; i<N; ++i) {
		AtlasIO::ChartIO chart;
		
		unsigned int M;
		in >> M;
		for (unsigned int j=0; j<M; ++j) {
			unsigned int vertex_id;
			in >> vertex_id;
			float x, y;
			in >> x >> y;
			chart[vertex_id] = std::make_pair(x, y);
		}
		
		atlas.Charts[i] = chart;
	}

	in >> N;
	atlas.Colors.resize(3*N);
	for (unsigned int i=0; i<N; ++i) {
        for (unsigned int j=0; j<3; ++j) {
            in >> atlas.Colors[3*i + j];            
        }
	}
	
	return in;
}

inline
void LoadAtlas(const std::string& file, AtlasIO& atlas) {
	std::ifstream input(file.c_str());
	input >> atlas;
	input.close();
}

inline
void WriteAtlas(const std::string& file, const AtlasIO& atlas) {
	std::ofstream output(file.c_str());
	output << atlas;
	output.close();
}

#endif
