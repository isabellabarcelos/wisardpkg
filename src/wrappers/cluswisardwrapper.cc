class ClusWisardWrapper: public ClusWisard{
public:
  ClusWisardWrapper(int addressSize, float minScore, int threshold, int discriminatorsLimit, int seed = 42, py::kwargs kwargs = py::kwargs())
  : ClusWisard(addressSize, minScore, threshold, discriminatorsLimit, seed)
{
  for(auto arg: kwargs){
      std::string key = std::string(py::str(arg.first));

      if(key == "seed") continue; // 🔥 importante

      if(key == "bleachingActivated")
        bleachingActivated = arg.second.cast<bool>();

      if(key == "verbose")
        verbose = arg.second.cast<bool>();

      if(key == "ignoreZero")
        ignoreZero = arg.second.cast<bool>();

      if(key == "completeAddressing")
        completeAddressing = arg.second.cast<bool>();

      if(key == "base")
        base = arg.second.cast<int>();

      if(key == "searchBestConfidence")
        searchBestConfidence = arg.second.cast<bool>();

      if(key == "returnConfidence")
        returnConfidence = arg.second.cast<bool>();

      if(key == "returnActivationDegree")
        returnActivationDegree = arg.second.cast<bool>();

      if(key == "returnClassesDegrees")
        returnClassesDegrees = arg.second.cast<bool>();

      if(key == "confidence")
        confidence = arg.second.cast<int>();
  }
}

  // 🔹 Classificação Python-friendly
  py::list pyClassify(const std::vector<std::vector<int>>& images){
    float numberOfRAMS = calculateNumberOfRams(images[0].size(), addressSize, completeAddressing);

    py::list labels(images.size());
    for(unsigned int i=0; i<images.size(); i++){
      if(verbose) std::cout << "\rclassifying " << i+1 << " of " << images.size();

      std::map<std::string,int> candidates = classify(images[i], searchBestConfidence);
      std::string label = Bleaching::getBiggestCandidate(candidates);
      std::string aClass = label.substr(0,label.find("::"));

      if(returnConfidence || returnActivationDegree || returnClassesDegrees){
        labels[i] = py::dict(py::arg("class")=aClass);
      }

      if(returnActivationDegree){
        float activationDegree = candidates[label]/numberOfRAMS;
        labels[i]["activationDegree"] = activationDegree;
      }

      if(returnConfidence){
        float conf = Bleaching::getConfidence(candidates, candidates[label]);
        labels[i]["confidence"] = conf;
      }

      if(returnClassesDegrees){
        labels[i]["classesDegrees"] = getClassesDegrees(candidates);
      }

      if(!returnActivationDegree && !returnConfidence && !returnClassesDegrees){
        labels[i] = aClass;
      }

      candidates.clear();
      std::map<std::string,int>().swap(candidates);
    }

    if(verbose) std::cout << "\r" << std::endl;
    return labels;
  }

protected:
  py::list getClassesDegrees(std::map<std::string, int> candidates) const{
    float total = 0;
    int index = 0;

    py::list output(candidates.size());

    for(auto& i : candidates)
      total += i.second;

    if(total == 0) total = 1;

    for(auto& i : candidates){
      std::string aClass = i.first.substr(0, i.first.find("::"));
      output[index] = py::dict(
        py::arg("class") = aClass,
        py::arg("degree") = (i.second / total)
      );
      index++;
    }

    return output;
  }
};