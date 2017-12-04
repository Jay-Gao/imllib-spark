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

import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.optimization.{Gradient, Optimizer}
import breeze.linalg.{DenseVector => BDV, Vector => BV}
import com.intel.imllib.util.vectorUtils.{fromBreeze, toBreeze}
import com.intel.imllib.optimization._
import com.intel.imllib.util.isConverged
import org.apache.log4j.Logger
import org.apache.spark.annotation.{DeveloperApi, Experimental}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

class FMGradient(val task: Int, val k0: Boolean, val k1: Boolean, val k2: Int,
								 val numFeatures: Int, val min: Double, val max: Double,
								 val r0: Double, val r1: Double, val r2: Double) extends Gradient {

	def this() = this(0, false, false, 4, -1, Double.MinValue, Double.MaxValue, 0, 0, 0)

	private def predict(data: Vector, weights: Vector): (Double, Array[Double]) = {
		var pred = if (k0) weights(weights.size - 1) else 0.0

		if (k1) {
			val pos = numFeatures * k2
			data.foreachActive {
				case (i, v) =>
					pred += weights(pos + i) * v
			}
		}

		val sum = Array.fill(k2)(0.0)
		for (f <- 0 until k2) {
			var sumSqr = 0.0
			data.foreachActive {
				case (i, v) =>
					val d = weights(i * k2 + f) * v
					sum(f) += d
					sumSqr += d * d
			}
			pred += (sum(f) * sum(f) - sumSqr) * 0.5
		}

		if (task == 0) {
			pred = Math.min(Math.max(pred, min), max)
		}

		(pred, sum)
	}

	override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
		val (pred, sum) = predict(data, weights)
		val mult = task match {
			case 0 =>
				pred - label
			case 1 =>
				-label * (1.0 - 1.0 / (1.0 + Math.exp(-label * pred)))
		}
		val len = weights.size
		val gradients = new Array[Double](len)
		if (k0) {
			gradients(len - 1) = mult + r0 * weights(len - 1)
		}
		if (k1) {
			val pos = numFeatures * k2
			data.foreachActive {
				case (i, v) =>
					gradients(pos + i) = v * mult + r1 * weights(pos + i)
			}
		}
		data.foreachActive {
			case (i, v) =>
				val pos = i * k2
				for ( f <- 0 until k2) {
					gradients(pos + f) = (sum(f) * v - weights(pos + f) * v * v) *  mult + r2 * weights(pos + f)
				}
		}
		val weights_ = toBreeze(weights)
		val regLoss0 = if (k0) r0 * weights_(-1) * weights_(-1) else 0
		val regLoss1 = if (k1) r1 * weights_(numFeatures * k2 until -1).dot(weights_(numFeatures * k2 until -1)) else 0
		val regLoss2 = 0.5 * r2 * weights_(0 until numFeatures * k2).dot(weights_(0 until numFeatures * k2))
		val regLoss = regLoss0 + regLoss1 + regLoss2
		val loss = task match {
			case 0 =>
				0.5 * (pred - label) * (pred - label) + regLoss
			case 1 =>
				math.log(1 + math.exp(-label * pred))
		}
		(Vectors.dense(gradients), loss)
	}

	override def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double = {
		throw new Exception("Not Defined.")
	}
}

class FMOptimizer(private var gradient: FMGradient,
									private var updater: Updater) extends Optimizer {

	private var numIterations: Int = 100
	private var miniBatchFraction: Double = 1.0
	private var convergenceTol: Double = 0.001
	this.updater.setRegParam(0)
	def setGradient(g: FMGradient): this.type = {
		this.gradient = g
		this
	}

	def setUpdater(updater: Updater): this.type = {
		this.updater = updater.setRegParam(0)
		this
	}

	def setMiniBatchFraction(fraction: Double): this.type = {
		require(fraction > 0 && fraction <= 1.0,
			s"Fraction for mini-batch SGD must be in range (0, 1] but got ${fraction}")
		this.miniBatchFraction = fraction
		this
	}

	/**
		* Set the number of iterations for SGD. Default 100.
		*/
	def setNumIterations(iters: Int): this.type = {
		require(iters >= 0,
			s"Number of iterations must be nonnegative but got ${iters}")
		this.numIterations = iters
		this
	}

	def setConvergenceTol(tolerance: Double): this.type = {
		require(tolerance >= 0.0 && tolerance <= 1.0,
			s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
		this.convergenceTol = tolerance
		this
	}

	@DeveloperApi
	def optimize(data: RDD[(Double, Vector)],
							 initialWeights: Vector): Vector = {
		val (weights, _) = FMOptimizer.runMiniBatch(data,
																								gradient,
																								updater,
																								numIterations,
																								miniBatchFraction,
																								initialWeights,
																								convergenceTol)
		weights
	}
}

object FMOptimizer {
	@transient lazy val log = Logger.getLogger(getClass.getName)
	def runMiniBatch(data: RDD[(Double, Vector)],
									 gradient: FMGradient,
									 updater: Updater,
									 numIterations: Int,
									 miniBatchFraction: Double,
									 initialWeights: Vector,
									 convergenceTol: Double): (Vector, Array[Double]) = {
		// convergenceTol should be set with non minibatch settings
		if (miniBatchFraction < 1.0 && convergenceTol > 0.0) {
			println("Testing against a convergenceTol when using miniBatchFraction " +
				"< 1.0 can be unstable because of the stochasticity in sampling.")
		}

		if (numIterations * miniBatchFraction < 1.0) {
			println("Not all examples will be used if numIterations * miniBatchFraction < 1.0: " +
				s"numIterations=$numIterations and miniBatchFraction=$miniBatchFraction")
		}

		val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
		// Record previous weight and current one to calculate solution vector difference

		var previousWeights: Option[Vector] = None
		var currentWeights: Option[Vector] = None

		val numExamples = data.count()

		// if no data, return initial weights to avoid NaNs
		if (numExamples == 0) {
			println("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
			return (initialWeights, stochasticLossHistory.toArray)
		}

		if (numExamples * miniBatchFraction < 1) {
			println("The miniBatchFraction is too small")
		}

		// Initialize weights as a column vector
		var weights = Vectors.dense(initialWeights.toArray)
		val n = weights.size
		val slices = data.getNumPartitions

		var updater_ = updater match {
			case _: SimpleUpdater =>
				updater.asInstanceOf[SimpleUpdater]
			case _: MomentumUpdater =>
				updater.asInstanceOf[MomentumUpdater].initializeMomentum(n)
			case _: AdagradUpdater =>
				updater.asInstanceOf[AdagradUpdater].initializeSquare(n)
			case _: RMSPropUpdater =>
				updater.asInstanceOf[RMSPropUpdater].initializeSquare(n)
			case _: AdamUpdater =>
				updater.asInstanceOf[AdamUpdater].initialMomentum(n).initialSquare(n)
		}

		var converged = false // indicates whether converged based on convergenceTol
		var i = 1
		// update weights with updater
		while (!converged && i <= numIterations) {
			val bcWeights = data.context.broadcast(weights)
			// Sample a subset (fraction miniBatchFraction) of the total data
			// compute and sum up the subgradients on this subset (this is one map-reduce)
			val (gradientSum, lossSum, miniBatchSize) = data.sample(false, miniBatchFraction, 42 + i)
				.treeAggregate((BDV.zeros[Double](n), 0.0, 0L))(
					seqOp = (c, v) => {
						// c: (grad, loss, count), v: (label, features)
						val (grad, loss) = gradient.compute(v._2, v._1, bcWeights.value)
						(c._1 + toBreeze(grad), c._2 + loss, c._3 + 1)
					},
					combOp = (c1, c2) => {
						// c: (grad, loss, count)
						(c1._1 + c2._1, c1._2 + c2._2, c1._3 + c2._3)
					})

			if (miniBatchSize > 0) {
				/**
					* lossSum is computed using the weights from the previous iteration
					* and regVal is the regularization value computed in the previous iteration as well.
					*/
				println(s"iter: $i, batch size: $miniBatchSize, train_avgloss: ${lossSum / miniBatchSize}")

				stochasticLossHistory += lossSum / miniBatchSize
				// compute updates.
				val update = updater_.compute(weights, fromBreeze(gradientSum / miniBatchSize.toDouble), i)
				weights = update._1

				previousWeights = currentWeights
				currentWeights = Some(weights)
				if (previousWeights.isDefined && currentWeights.isDefined) {
					converged = isConverged(previousWeights.get,
						currentWeights.get, convergenceTol)
				}
			} else {
				log.warn(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
			}
			i += 1
		}
		(weights, stochasticLossHistory.toArray)
	}
}