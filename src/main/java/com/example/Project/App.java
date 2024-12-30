package com.example.Project;

import java.io.File;
import java.util.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.attribute.StringToNominal;

public class App {

    public static Instances loadDataset(String filePath) {
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(filePath));
            Instances data = loader.getDataSet();
            System.out.println("Data loaded successfully from CSV.");
            return data;
        } catch (Exception e) {
            System.err.println("Error loading dataset: " + e.getMessage());
            return null;
        }
    }

    public static Instances handleMissingValues(Instances data) {
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).isNumeric()) {
                double mean = data.meanOrMode(i);
                for (int j = 0; j < data.numInstances(); j++) {
                    if (data.instance(j).isMissing(i)) {
                        data.instance(j).setValue(i, mean);
                    }
                }
            } else {
                String mode = getMode(data, i);
                for (int j = 0; j < data.numInstances(); j++) {
                    if (data.instance(j).isMissing(i)) {
                        data.instance(j).setValue(i, mode);
                    }
                }
            }
        }
        System.out.println("Missing values handled.");
        return data;
    }

    private static String getMode(Instances data, int attributeIndex) {
        Map<String, Long> frequencyMap = new HashMap<>();
        for (int i = 0; i < data.numInstances(); i++) {
            if (!data.instance(i).isMissing(attributeIndex)) {
                String value = data.instance(i).stringValue(attributeIndex);
                frequencyMap.put(value, frequencyMap.getOrDefault(value, 0L) + 1);
            }
        }
        return Collections.max(frequencyMap.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    public static Instances encodeCategoricalFeatures(Instances data) {
        try {
            StringToNominal stringToNominal = new StringToNominal();
            stringToNominal.setAttributeRange("1-last");
            stringToNominal.setInputFormat(data);
            data = Filter.useFilter(data, stringToNominal);
            System.out.println("Categorical features encoded.");
        } catch (Exception e) {
            System.err.println("Error encoding categorical features: " + e.getMessage());
        }
        return data;
    }

    public static Instances scaleFeatures(Instances data) {
        try {
            Standardize standardize = new Standardize();
            standardize.setInputFormat(data);
            data = Filter.useFilter(data, standardize);
            System.out.println("Features scaled.");
        } catch (Exception e) {
            System.err.println("Error scaling features: " + e.getMessage());
        }
        return data;
    }

    public static void printData(Instances data) {
        System.out.println("Printing Preprocessed Data:");
        for (int i = 0; i < data.numInstances(); i++) {
            System.out.println(data.instance(i));
        }
    }

    public static void main(String[] args) {
        String filePath = "C:\\Users\\mohan\\Downloads\\Mentalhealth_Dataset.csv";
        Instances data = loadDataset(filePath);
        if (data != null) {
            data = handleMissingValues(data);
            printData(data); 
            data = encodeCategoricalFeatures(data);
            printData(data); 

            data = scaleFeatures(data);
            printData(data); 

            System.out.println("Preprocessing complete. Ready for the next steps.");
        }
    }
}
