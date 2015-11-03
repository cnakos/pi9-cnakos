import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;

import org.apache.uima.cas.CAS;
import org.apache.uima.cas.CASException;
import org.apache.uima.collection.CasConsumer_ImplBase;
import org.apache.uima.collection.CollectionException;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.resource.ResourceProcessException;

import rank.CompositeRanker;
import rank.IRanker;
import rank.NLPUtils;
import rank.NgramRanker;
import rank.OtherRanker;
import type.Measurement;
import type.Passage;
import type.Question;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.Logistic;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;

/**
 * This CAS Consumer generates the report file with the method metrics
 */
public class PassageRankingWriter extends CasConsumer_ImplBase {
	final String PARAM_OUTPUTDIR = "OutputDir";

	final String PARAM_NUMFOLDS = "NumFolds";

	final String OUTPUT_FILENAME = "RankMeasurements.csv";

	File mOutputDir;

	int numFolds;

	IRanker ngramRanker, otherRanker;

	CompositeRanker compositeRanker;
	
	// By the typical definition of logistic regression, this should stay 0.5.
	double mScoreThreshold = 0.5;
	
	FastVector featureVector;

	@Override
	public void initialize() throws ResourceInitializationException {
		String mOutputDirStr = (String) getConfigParameterValue(PARAM_OUTPUTDIR);
		if (mOutputDirStr != null) {
			mOutputDir = new File(mOutputDirStr);
			if (!mOutputDir.exists()) {
				mOutputDir.mkdirs();
			}
		}
		numFolds = (int) getConfigParameterValue(PARAM_NUMFOLDS);
		// Parameter-based mScoreThreshold assignment goes here if it ever needs adjusting.
		
		// Set up feature vector for regression.
		// Score attributes.
		Attribute ngramScore = new Attribute("ngramScore");
		Attribute otherScore = new Attribute("otherScore");
		
		// Class attribute for relevance.
		FastVector classVector = new FastVector(2);
		classVector.addElement("relevant");
		classVector.addElement("irrelevant");
		Attribute relevance = new Attribute("relevance", classVector);
		
		// Get the feature vector.
		featureVector = new FastVector(3);
		featureVector.addElement(ngramScore);
		featureVector.addElement(otherScore);
		featureVector.addElement(relevance);
	}

	@Override
	public void processCas(CAS arg0) throws ResourceProcessException {
		System.out.println(">> Passage Ranking Writer Processing");
		// Import the CAS as a aJCas
		JCas aJCas = null;
		File outputFile = null;
		PrintWriter writer = null;
		try {
			aJCas = arg0.getJCas();
			
			// Initialize rankers here so we can calculate IDF for Okapi BM25.
			compositeRanker = new CompositeRanker();
			ngramRanker = new NgramRanker();
			otherRanker = new OtherRanker(aJCas, 1.6, 0.75);
			compositeRanker.addRanker(ngramRanker);
			compositeRanker.addRanker(otherRanker);
      
			try {
				outputFile = new File(Paths.get(mOutputDir.getAbsolutePath(), OUTPUT_FILENAME).toString());
				outputFile.getParentFile().mkdirs();
				writer = new PrintWriter(outputFile);
			} catch (FileNotFoundException e) {
				System.out.printf("Output file could not be written: %s\n",
						Paths.get(mOutputDir.getAbsolutePath(), OUTPUT_FILENAME).toString());
				return;
			}

			writer.println("question_id,p_at_1,p_at_5,rr,ap");

			// Retrieve all the questions.
			List<Question> allQuestions = UimaUtils.getAnnotations(aJCas, Question.class);

			// Train a model to get the optimized weights.
			List<IRanker> rankers = new ArrayList<IRanker>();
			rankers.add(ngramRanker);
			rankers.add(otherRanker);
			List<Double> weights = train(allQuestions, rankers);

			// Use the learned weights for rank aggregation (composition).
			compositeRanker.setWeights(weights);

			// TODO: Here one needs to sort the questions in ascending order of their question ID
			Comparator<Question> qComparator = new Comparator<Question>() {
				public int compare(Question o1, Question o2) {
					return o1.getId().compareTo(o2.getId());
				}
			};
			Collections.sort(allQuestions, qComparator);
			
			double map = 0.0;
			double mrr = 0.0;
			
			for (Question question : allQuestions) {
				List<Passage> passages = UimaUtils.convertFSListToList(question.getPassages(),
						Passage.class);

				// You could compare the composite ranker with individual rankers her.
				// List<Passage> ngramRankedPassages = ngramRanker.rank(question, passages);
				// List<Passage> otherRankedPassages = otherRanker.rank(question, passages);
				List<Passage> compositeRankedPassages = compositeRanker.rank(question, passages);
				
				// TODO: Compute the measurement for this question.
				int tp = 0, fp = 0, fn = 0, tn = 0;
				double p = 0.0, ap = 0.0, rr = 0.0;
				double p_at_1 = 0.0, p_at_5 = 0.0;
				for (int i = 0; i < compositeRankedPassages.size(); i++) {
					// Not the most efficient thing in the world, but not worth refactoring IRanker to fix.
					double score = compositeRanker.score(question,  compositeRankedPassages.get(i));
					boolean compositeLabel = score > mScoreThreshold;
					boolean trueLabel = compositeRankedPassages.get(i).getLabel();
					if (trueLabel) {
						p += 1.0;
						ap += p / (i + 1.0);
						if (rr == 0.0)
							rr = 1.0 / (i + 1.0);
						if (i == 0)
							p_at_1 += 1.0;
						if (i < 5)
							p_at_5 += 0.2; // Fixed n, so no point dividing it later.
					}
					// 
					if (compositeLabel && trueLabel)
						tp++;
					else if (compositeLabel && !trueLabel)
						fp++;
					else if (!compositeLabel && trueLabel)
						fn++;
					else
						tn++;
				}
				if (p > 0.0)
					ap /= p;
				double precision = tp + fp > 0.0 ? ((double) tp) / (tp + fp) : 0.0;
				double recall = tp + fn > 0.0 ? ((double) tp) / (tp + fn) : 0.0;
				double f1 = precision + recall > 0.0 ? 2 * precision * recall /(precision + recall) : 0.0;
				
				mrr += rr;
				map += ap;

				// Mostly pointless at this stage, but I'll go with the template and set m.
				// This would make sense in a separate AE, but having the ranker here is nicer.
				Measurement m = new Measurement(aJCas);
				m.setPrecisionAt1(p_at_1);
				m.setPrecisionAt5(p_at_5);
				m.setReciprocalRank(rr);
				m.setAveragePrecision(ap);
				question.setMeasurement(m);

				writer.printf("%s,%.3f,%.3f,%.3f,%.3f\n", question.getId(), m.getPrecisionAt1(),
						m.getPrecisionAt5(), m.getReciprocalRank(), m.getAveragePrecision());
			}
			
			mrr /= allQuestions.size();
			map /= allQuestions.size();
			System.out.println("MRR: " + Double.toString(mrr) + ", MAP: " + Double.toString(map));
		} catch (CASException e) {
			try {
				throw new CollectionException(e);
			} catch (CollectionException e1) {
				e1.printStackTrace();
			}
		} finally {
			if (writer != null)
				writer.close();
		}
	}

	/**
	 * Trains a logistic regression model on the given question data through cross validation.
	 * Optimizes the weights on individual rankers with respect to P@1.
	 * 
	 * @param allQuestions
	 * @param rankers
	 * @return the optimized weights
	 */
	public List<Double> train(List<Question> questions, List<IRanker> rankers) {
		// TODO: Complete the implementation of this method.
		System.out.println(String.format(
				">> Training a logistic regression model with %d-fold cross validation", numFolds));

		// ### Implementation guidelines ###
		// 1. Split the data into k folds for cross validation.
		// 2. For each fold, do the following:
		// 2-1. Get the training and validation datasets for this fold.
		// 2-2. Train the model on the training dataset while tuning hyperparameters.
		// 2-3. Compute P@1 of the trained composite model on the validation dataset.
		// 3. Compute the average of P@1 over the validation datasets.
		// 4. Get the best hyperparameters that give you the best average of P@1.
		// 5. Train the model on the entire dataset with the best hyperparameters you get in step 4.
		// 6. Return the learned weights you get in step 5.

		// weights[0] is for the n-gram ranker, and weights[1] is for the other ranker.
		List<Double> weights = new ArrayList<Double>();
		
		// 1. Split the data into k folds for cross validation.
		// folds[0] is the 1st fold, and folds[1] is the 2nd one, etc.
		List<List<Question>> folds = new ArrayList<List<Question>>();
		int foldSize = (int) Math.ceil(((double) questions.size()) / numFolds);
		List<Question> remainingQuestions = new ArrayList<Question>();
		for (Question question : questions) {
			remainingQuestions.add(question);
		}
		while (remainingQuestions.size() > foldSize) {
			List<Question> fold = RandomUtils.getRandomSubset(remainingQuestions, foldSize);
			folds.add(fold);
			remainingQuestions.removeAll(fold);
		}
		folds.add(remainingQuestions);

		Instances allData = getEmptyInstances();
		List<Instances> validationData = new ArrayList<Instances>();
		List<Instances> trainingData = new ArrayList<Instances>();
		
		// 2. For each fold, do the following:
		// 2-1. Get the training and validation datasets for this fold.
		for (int i = 0; i < folds.size(); i++) {
			validationData.add(convertQuestionsToInstances(folds.get(i), rankers));
		}
		for (int i = 0; i < folds.size(); i++) {
			// Get empty Instances for the training data.
			Instances data = getEmptyInstances();
			for (int j = 0; j < folds.size(); j++)
				if (i != j)
					data.addAll(validationData.get(j));
			trainingData.add(data);
			allData.addAll(data);
		}
		
		// 2-2. Train the model on the training dataset while tuning hyperparameters.
		double[] ridgeValues = {0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0}; 
				// 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0};
		double[] ridgePrecisions = new double[ridgeValues.length];
		for (int i = 0; i < ridgeValues.length; i++) {
			double precisionSum = 0.0;
			for (int j = 0; j < folds.size(); j++) {
				try {
					Logistic classifier = new Logistic();
					classifier.setRidge(ridgeValues[i]);
					classifier.buildClassifier(trainingData.get(j));
					
					// WEKA evaluation not strictly necessary, since P@1 is manual.
					Evaluation evaluation = new Evaluation(trainingData.get(j));
					evaluation.evaluateModel(classifier, validationData.get(j));
					//System.out.println(evaluation.toSummaryString());
					//System.out.println(classifier.toString());
					
					// 2-3. Compute P@1 of the trained composite model on the validation dataset.
					//System.out.println(classifier.toString());
					List<Double> foldWeights = getWeightsFromClassifier(classifier);
					double p_at_1 = getAveragePAt1(folds.get(j), rankers, foldWeights);
					precisionSum += p_at_1;
				} catch (Exception e) {
					// Eclipse doesn't know what kind of Exception buildClassifier() throws.
					e.printStackTrace();
					return null;
				}
			}
			// 3. Compute the average of P@1 over the validation datasets.
			ridgePrecisions[i] = precisionSum / folds.size();
		}
		
		// 4. Get the best hyperparameters that give you the best average of P@1.
		int bestIndex = 0;
		double bestPrecision = 0.0;
		for (int i = 0; i < ridgeValues.length; i++) {
			System.out.println(ridgeValues[i]);
			System.out.println(ridgePrecisions[i]);
			if (ridgePrecisions[i] > bestPrecision) {
				bestIndex = i;
				bestPrecision = ridgePrecisions[i];
			}
		}
		
		// 5. Train the model on the entire dataset with the best hyperparameters you get in step 4.
		try {
			Logistic overall = new Logistic();
			overall.setRidge(ridgeValues[bestIndex]);
			overall.buildClassifier(allData);
			// 6. Return the learned weights you get in step 5.
			return getWeightsFromClassifier(overall);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * Converts the given questions to Weka instances.
	 * 
	 * @param questions
	 * @param rankers
	 * @return Weka instances
	 */
	public Instances convertQuestionsToInstances(List<Question> questions, List<IRanker> rankers) {
		Instances data = getEmptyInstances();
		
		for (Question question : questions) {
			for (Passage passage : UimaUtils.convertFSListToList(question.getPassages(), Passage.class)) {
				Instance instance = new DenseInstance(3);
				instance.setValue((Attribute) featureVector.elementAt(0), rankers.get(0).score(question, passage));
				instance.setValue((Attribute) featureVector.elementAt(1), rankers.get(1).score(question, passage));
				instance.setValue((Attribute) featureVector.elementAt(2), passage.getLabel() ? "relevant" : "irrelevant");
				data.add(instance);
			}
		}
		
		return data;
	}
	
	// Convenience method for getting an emtpy Instances to fill.
	public Instances getEmptyInstances() {
		Instances data = new Instances("Rel", featureVector, 0);
		data.setClassIndex(2);
		return data;
	}
	
	// Return the average P@1 for a given test set.
	public double getAveragePAt1(List<Question> questions, List<IRanker> rankers, List<Double> weights) {
		// There's something seriously broken with this approach if this method has to be this complicated.
		// For instance, we have to recompute all scores because it's easier than pulling them back out of Instances.
		CompositeRanker composite = new CompositeRanker();
		for (IRanker ranker : rankers)
			composite.addRanker(ranker);
		composite.setWeights(weights);
		int pIndex = 0;
		double total = 0.0;
		for (Question question : questions) {
			
			Passage bestPassage = null;
			double bestScore = 0.0;
			List<Passage> passageList = UimaUtils.convertFSListToList(question.getPassages(), Passage.class); 
			for (Passage passage : passageList) {
				double score = composite.score(question, passage);
				if (score > bestScore) {
					bestScore = score;
					bestPassage = passage;
				}
				pIndex++;
			}
			if (bestPassage.getLabel())
				total += 1.0;
		}
		return total / questions.size();
	}
	
	// Return the weights found by the classifier.
	public List<Double> getWeightsFromClassifier(Logistic classifier) {
		List<Double> weights = new ArrayList<Double>();
		weights.add(classifier.coefficients()[1][0]);
		weights.add(classifier.coefficients()[2][0]);
		return weights;
	}
}
