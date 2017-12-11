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

import org.apache.spark.rdd.RDD
import breeze.linalg.{Matrix=>BM, sum, CSCMatrix => BSM, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}
import breeze.linalg.VectorBuilder
import com.intel.imllib.optimization._
import com.intel.imllib.util._

import scala.collection.mutable.ArrayBuffer


class FFMGradient(task: Int,
									numFields: Int,
									dim: (Boolean, Boolean, Int),
									regParams: (Double, Double, Double),
									minValue: Double,
									maxValue: Double) extends Serializable {
	private val f = numFields
	private val (k0, k1, k2) = dim
	private val (r0, r1, r2) = regParams

	def calculateScore(feature: Array[(Int, Int, Double)],
						weights: FFM_DENSE_PARAM): Double = {
		val (w, v) = weights
		// bias.
		var score = if (k0) w(-1) else 0
		// first-order weights.
		if (k1) {
			feature.foreach {
				// (field, feature, value)
				case (_, j, x) =>
					score += w(j) * x
			}
		}
		// second-order weights.
		for (p <- 0 until feature.length - 1) {
			// (field, feature, value)
			val (f1, j1, x1) = feature(p)
			for (q <- p + 1 until feature.length) {
				val (f2, j2, x2) = feature(q)
				score += v(::, j1 * f + f2) .dot(v(::, j2 * f + f1)) * x1 * x2
			}
		}
		task match {
			case 0 => score = math.max(math.min(minValue, score), maxValue)
			case 1 =>
		}

		score
	}

	def compute(data: (Double, Array[(Int, Int, Double)]),
							weights: FFM_DENSE_PARAM,
							cumGradients: FFM_DENSE_PARAM,
							cumCounter: FFM_DENSE_PARAM): Double = {
		val (label, feature) = data
		val (w, v) = weights
		val score: Double = calculateScore(feature, weights)
		val kappa = if (task == 0) score - label else -label / (1 + math.exp(label * score))
		val (wGradients, vGradients) = cumGradients
		val (wCounter, vCounter) = cumCounter
//		val wGradients = BDV.zeros[Double](w.length)
//		val wCounter = BDV.zeros[Double](w.length)
////		val vGradients = new BSM.Builder[Double](v.rows, v.cols)
////		val vCounter = BSM.zeros[Double](v.rows, v.cols)
//		val vGradients = BDM.zeros[Double](v.rows, v.cols)
//		val vCounter = BDM.zeros[Double](v.rows, v.cols)

		var loss: Double = 0

		if (k0) {
			wGradients(-1) += kappa + r0 * w(-1)
			wCounter(-1) += 1.0
			loss += r0 * w(-1) * w(-1)
		}
		if (k1) {
			feature.foreach {
				case (_, j, x) =>
					wGradients(j) += kappa * x + r0 * w(j)
					wCounter(j) += 1.0
			}
			loss += r1 * w(0 to -1).dot(w(0 to -1))
		}

//		for (p <- 0 until feature.length) {
//			val (f1, j1, x1) = feature(p)
//			for (q <- p + 1 until feature.length) {
//				val (f2, j2, x2) = feature(q)
//				val g_j1_f2 = v(j2 * f + f1, ::) * kappa * x1 * x2 + v(j1 * f + f2, ::) * r2
//				val g_j2_f1 = v(j1 * f + f2, ::) * kappa * x1 * x2 + v(j2 * f + f1, ::) * r2
//				for (k <- 0 until k2) {
//					vGradients.add(j1 * f + f2, k, g_j1_f2(k))
//					vGradients.add(j2 * f + f1, k, g_j2_f1(k))
//					vCounter(j1 * f + f2, k) = 1.0
//					vCounter(j2 * f + f1, k) = 1.0
//				}
//			}
//		}
		var _vCountIndex = Set[Int]()
		for (p <- 0 until feature.length) {
			val (f1, j1, x1) = feature(p)
			for (q <- p + 1 until feature.length) {
				val (f2, j2, x2) = feature(q)
				vGradients(::, j1 * f + f2) :+= v(::, j2 * f + f1) * kappa * x1 * x2 + v(::, j1 * f + f2) * r2
				vGradients(::, j2 * f + f1) :+= v(::, j1 * f + f2) * kappa * x1 * x2 + v(::, j2 * f + f1) * r2
				_vCountIndex += (j1 * f + f2, j2 * f + f1)
//				vCounter(::, j1 * f + f2) := 1.0
//				vCounter(::, j2 * f + f1) := 1.0
			}
		}
		// update counter
		_vCountIndex.foreach(i => vCounter(::, i) :+= 1.0)

		loss += r2 * sum(v :* v)

		task match {
			case 0 => loss += 0.5 * (score - label) * (score - label)
			case 1 => loss +=  math.log(1 + math.exp(-label * score))
		}

		loss
	}
}

class FFMOptimizer() extends Serializable {

	private var numIterations: Int = 10
	private var miniBatchFraction: Double = 1.0
	private var convergenceTol: Double = 0.001
	private var gradient: FFMGradient = _
	private var updater: Updater = new AdagradUpdater()

	def setGradient(g: FFMGradient): this.type = {
		this.gradient = g
		this
	}

	def setUpdater(updater: Updater): this.type = {
		this.updater = updater
		this
	}

	def setNumIterations(i: Int): this.type = {
		require(i > 0)
		this.numIterations = i
		this
	}

	def setMiniBatchFraction(fraction: Double): this.type = {
		require(fraction > 0 && fraction <= 1.0)
		this.miniBatchFraction = fraction
		this
	}

	def setConvergenceTol(c: Double): this.type = {
		this.convergenceTol = c
		this
	}

	def optimize(data: RDD[(Double, Array[(Int, Int, Double)])],
							 initialWeights: FFM_DENSE_PARAM
							): FFM_DENSE_PARAM = {
		// regularization parameters should be set outside updater.
		this.updater.setRegParam(0)
		val (weights, _) = this.runMiniBatch(data, initialWeights)
		weights

	}

	def runMiniBatch(data: RDD[(Double, Array[(Int, Int, Double)])],
									 initialWeights: FFM_DENSE_PARAM): (FFM_DENSE_PARAM, Array[Double]) = {
		val lossHistory = new ArrayBuffer[Double]()
		var (w, v) = initialWeights
		val (rows, cols) = (v.rows, v.cols)
		val len = w.length
		val sizes = len match {
			case 0 => Array[Int](rows * cols)
			case _ => Array[Int](w.length, rows * cols)
		}

		this.updater.setRegParam(0).initialize(sizes: _*)
		var converged = false
		var iter = 1
		while (!converged && iter < this.numIterations) {
			val bw = data.context.broadcast(w)
			val bv = data.context.broadcast(v)
			val (gradientsSum, counter, loss, miniBatchSize) = data.sample(false, this.miniBatchFraction, 64 + iter)
		  	.treeAggregate((BDV.zeros[Double](w.length), BDM.zeros[Double](rows, cols)),
					(BDV.zeros[Double](w.length), BDM.zeros[Double](rows, cols)), 0.0, 0L)(
					seqOp = (x, y) => {
						val l = this.gradient.compute(y, (bw.value, bv.value), x._1, x._2)
						(x._1, x._2, x._3 + l, x._4 + 1)
					},
					combOp = (x, y) => {
						((x._1._1 + y._1._1, x._1._2 + y._1._2), (x._2._1 + y._2._1, x._2._2 + y._2._2), x._3 + y._3, x._4 + y._4)
					}
				)
			if (miniBatchSize > 0) {
				println(s"iter: $iter, batch size: $miniBatchSize, train_avgloss: ${loss / miniBatchSize}")
				lossHistory += loss / miniBatchSize
				// update parameters.
				len match {
					case 0 =>
						val (_, vg) = gradientsSum
						val (_, vc) = counter
						val flattenV = v.flatten().toDenseVector
						val meanFlattenG = vg.flatten().toDenseVector / (vc.flatten().toDenseVector + 1e-4)
						val (paramNew, _) = this.updater.compute(Array(flattenV), Array(meanFlattenG), iter)
						v = paramNew(0).toDenseVector.toDenseMatrix.reshape(rows, cols)
					case _ =>
						val (wg, vg) = gradientsSum
						val (wc, vc) = counter
						val flattenV = v.flatten().toDenseVector
						val meanFlattenG = vg.flatten().toDenseVector / (vc.flatten().toDenseVector + 1e-4)
						val meanWg = wg.toDenseVector / (wc.toDenseVector + 1e-4)
						val (paramNew, _) = this.updater.compute(Array(w, flattenV), Array(meanWg, meanFlattenG), iter)
						w = paramNew(0).toDenseVector
						v = paramNew(1).toDenseVector.toDenseMatrix.reshape(rows, cols)
				}
			}
//			converged = isConverged()
			iter += 1
		}
		((w, v), lossHistory.toArray)
	}
}
