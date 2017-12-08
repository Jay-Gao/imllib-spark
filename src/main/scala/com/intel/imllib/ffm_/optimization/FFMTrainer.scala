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

package com.intel.imllib.ffm_.optimization

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.stats.distributions.Gaussian
import com.intel.imllib.ffm_.FFMModel_
import com.intel.imllib.util._
import org.apache.spark.rdd.RDD

class FFMTrainer(private var task: Int,
								 private var numFeatures: Int,
								 private var numFields: Int,
								 private var dim: (Boolean, Boolean, Int),
								 private var regParams: (Double, Double, Double)) {
	private var k0: Boolean = dim._1
	private var k1: Boolean = dim._2
	private var k2: Int = dim._3
	private var r0: Double = regParams._1
	private var r1: Double = regParams._2
	private var r2: Double = regParams._3
	private var initMean: Double = 0
	private var initStd: Double = 0.01
	private var minLabel: Double = Double.MaxValue
	private var maxLabel: Double = Double.MinValue
	val optimizer = new FFMOptimizer()

	private def initializeWeights(): FFM_DENSE_PARAM = {
		require(this.numFeatures > 0)
		val w = (k0, k1) match {
			case (false, false) => BDV.zeros[Double](0)
			case (true, false)  => BDV.zeros[Double](1)
			case (true, true)   => BDV.zeros[Double](1 + this.numFeatures)
			case (false, true)  => BDV.zeros[Double](this.numFeatures)
		}
		val v = BDM.rand(this.k2, this.numFeatures * this.numFields, Gaussian(this.initMean, this.initStd))
		(w, v)
	}

	private def createModel(weights: FFM_DENSE_PARAM): FFMModel_ = {

		val (w, v) = weights
		val bias = if (k0) Some(w(-1)) else None
		val w_ = if (k1) Some(w(0 until w.length)) else None
		new FFMModel_(task, this.numFeatures, this.numFields, (bias, w_, v), minLabel, maxLabel)
	}

	def train(data: RDD[(Double, Array[(Int, Int, Double)])]): FFMModel_ = {
		val newData = task match {
			case 0 => data
			case 1 => data.map {case (l, feature) => (if (l > 0) 1.0 else -1.0, feature)}
		}
		val gradient = new FFMGradient(task, this.numFields, (k0, k1, k2), (r0, r1, r2), minLabel, maxLabel)
		this.optimizer.setGradient(gradient)
		val initialWeights = this.initializeWeights()
		val weights = this.optimizer.optimize(newData, initialWeights)
		createModel(weights)
	}
}