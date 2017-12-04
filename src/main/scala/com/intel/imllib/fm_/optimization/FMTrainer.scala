/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.imllib.fm_.optimization

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{DenseMatrix, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint

import com.intel.imllib.fm.regression.FMModel
import com.intel.imllib.optimization._

import scala.util.Random

class FMTrainer(private var task: Int,
								private var dim: (Boolean, Boolean, Int),
								private var regParams: (Double, Double, Double)) {

	def this() = this(0, (true, true, 8), (0, 1e-3, 1e-4))
	private var k0: Boolean = dim._1
	private var k1: Boolean = dim._2
	private var k2: Int = dim._3
	private var r0: Double = regParams._1
	private var r1: Double = regParams._2
	private var r2: Double = regParams._3
	private var initMean: Double = 0
	private var initStd: Double = 0.01
	private var numFeatures: Int = -1
	private var minLabel: Double = Double.MaxValue
	private var maxLabel: Double = Double.MinValue
	val optimizer = new FMOptimizer(new FMGradient(), new SimpleUpdater(0.01))
	/**
		* A (Boolean,Boolean,Int) 3-Tuple stands for whether the global bias term should be used, whether the one-way
		* interactions should be used, and the number of factors that are used for pairwise interactions, respectively.
		*/
	def setDim(dim: (Boolean, Boolean, Int)): this.type = {
		require(dim._3 > 0)
		this.k0 = dim._1
		this.k1 = dim._2
		this.k2 = dim._3
		this
	}

	/**
		*
		* @param addIntercept determines if the global bias term w0 should be used
		* @param add1Way determines if one-way interactions (bias terms for each variable)
		* @param numFactors the number of factors that are used for pairwise interactions
		*/
	def setDim(addIntercept: Boolean = true, add1Way: Boolean = true, numFactors: Int = 8): this.type = {
		setDim((addIntercept, add1Way, numFactors))
	}

	/**
		* @param regParams A (Double,Double,Double) 3-Tuple stands for the regularization parameters of intercept, one-way
		*                  interactions and pairwise interactions, respectively.
		*/
	def setRegParam(regParams: (Double, Double, Double)): this.type = {
		require(regParams._1 >= 0 && regParams._2 >= 0 && regParams._3 >= 0)
		this.r0 = regParams._1
		this.r1 = regParams._2
		this.r2 = regParams._3
		this
	}

	/**
		* @param regIntercept intercept regularization
		* @param reg1Way one-way interactions regularization
		* @param reg2Way pairwise interactions regularization
		*/
	def setRegParam(regIntercept: Double = 0, reg1Way: Double = 0, reg2Way: Double = 0): this.type = {
		setRegParam((regIntercept, reg1Way, reg2Way))
	}

	/**
		* @param initStd Standard Deviation used for factorization matrix initialization.
		*/
	def setInitStd(initStd: Double): this.type = {
		require(initStd > 0)
		this.initStd = initStd
		this
	}

	/**
		* Encode the FMModel to a dense vector, with its first numFeatures * numFactors elements representing the
		* factorization matrix v, sequential numFeaturs elements representing the one-way interactions weights w if k1 is
		* set to true, and the last element representing the intercept w0 if k0 is set to true.
		* The factorization matrix v is initialized by Gaussinan(0, initStd).
		* v : numFeatures * numFactors + w : [numFeatures] + w0 : [1]
		*/
	private def initializeWeights(): Vector = {
		(k0, k1) match {
			case (true, true) =>
				Vectors.dense(Array.fill(numFeatures * k2)(Random.nextGaussian() * initStd + initMean) ++
					Array.fill(numFeatures + 1)(0.0))

			case (true, false) =>
				Vectors.dense(Array.fill(numFeatures * k2)(Random.nextGaussian() * initStd + initMean) ++
					Array(0.0))

			case (false, true) =>
				Vectors.dense(Array.fill(numFeatures * k2)(Random.nextGaussian() * initStd + initMean) ++
					Array.fill(numFeatures)(0.0))

			case (false, false) =>
				Vectors.dense(Array.fill(numFeatures * k2)(Random.nextGaussian() * initStd + initMean))
		}
	}

	/**
		* Create a FMModle from an encoded vector.
		*/
	private def createModel(weights: Vector): FMModel = {

		val values = weights.toArray

		val v = new DenseMatrix(k2, numFeatures, values.slice(0, numFeatures * k2))

		val w = if (k1) Some(Vectors.dense(values.slice(numFeatures * k2, numFeatures * k2 + numFeatures))) else None

		val w0 = if (k0) values.last else 0.0

		new FMModel(task, v, w, w0, minLabel, maxLabel)
	}

	def train(data: RDD[LabeledPoint]): FMModel = {
		val newData = task match {
			case 0 => data.map(l => (l.label, l.features))
			case 1 => data.map(l => (if (l.label > 0) 1.0 else -1.0, l.features))
		}
		this.numFeatures = data.first().features.size
		val gradient = new FMGradient(task, k0, k1, k2, numFeatures, minLabel, maxLabel, r0, r1, r2)
		this.optimizer.setGradient(gradient)
		val initialWeights = this.initializeWeights()
		require(initialWeights.size > 246, s"but got ${initialWeights.size}")
		val weights = this.optimizer.optimize(newData, initialWeights)
		createModel(weights)
	}

}

