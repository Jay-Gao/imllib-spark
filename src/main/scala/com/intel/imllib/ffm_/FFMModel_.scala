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

package com.intel.imllib.ffm_

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, RegressionMetrics}
import breeze.linalg.{DenseVector=>BDV, DenseMatrix=>BDM, Vector=>BV, Matrix=>BM}
import com.intel.imllib.util.Saveable


class FFMModel_(task: Int,
								numFeatures: Int,
								numFields: Int,
								weights: (Option[Double], Option[BDV[Double]], BDM[Double]),
								minValue: Double,
								maxValue: Double
								) extends Serializable {
	private val n = numFeatures
	private val f = numFields
	private val (bias, weightsVector, factorsMatrix) = weights
	require(n > 0 && f > 0)
	require(n * f == factorsMatrix.cols)

	def predict(data: Array[(Int, Int, Double)]): Double = {
		var score = bias match {
			case Some(x) => x
			case _ => 0
		}
		weightsVector match {
			case Some(w) =>
				data.foreach {
					case (_, j, v) =>
						score += w(j) * v
				}
			case _ =>
		}
		for (p <- 0 until data.length - 1) {
			// (field, feature, value)
			val (f1, j1, x1) = data(p)
			for (q <- p + 1 until data.length) {
				val (f2, j2, x2) = data(q)
				score += factorsMatrix(::, j1 * f + f2).dot(factorsMatrix(::, j2 * f + f1)) * x1 * x2
			}
		}
		val pred = task match {
			case 0 =>
				math.max(math.min(minValue, score), maxValue)
			case 1 =>
				1 / (1 + math.exp(-score))
		}
		pred
	}

	def predict(data: RDD[Array[(Int, Int, Double)]]): RDD[Double] = {
		data.map(x => predict(x))
	}

	def evaluate(data: RDD[(Double, Array[(Int, Int, Double)])]): Map[String, Double] = {
		val predAndLabels = data.map(x => predict(x._2)).zip(data.map(x => if(x._1 > 0) 1.0 else 0))
		task match {
			case 0 =>
				val metrics = new RegressionMetrics(predAndLabels)
				Map(
					"mse" -> metrics.meanSquaredError,
					"mae" -> metrics.meanAbsoluteError,
					"r2" -> metrics.r2
				)
			case 1 =>
				val metrics = new BinaryClassificationMetrics(predAndLabels)
				val logLoss = predAndLabels.map {case (prob, label) => -label * math.log(prob) - (1 - label) * math.log(1 - prob)}.mean
				val acc = predAndLabels
					.map {case (prob, label) => (if(prob > 0.5) 1 else 0, label)}
			  	.map {case (pred, label) => if(pred == label) 1 else 0}.mean
				Map(
					"auc" -> metrics.areaUnderROC,
					"pr" -> metrics.areaUnderPR,
					"logLoss" -> logLoss,
					"acc" -> acc
				)
		}
	}
}

object FFMModel_ {

}
