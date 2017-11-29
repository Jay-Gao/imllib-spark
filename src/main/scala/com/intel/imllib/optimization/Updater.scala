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

package com.intel.imllib.optimization

import scala.math._
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import breeze.linalg.{Vector=>BV, DenseVector=>BDV}
import com.intel.imllib.util.vectorUtils._


abstract class Updater(
							 private var learningRate: Double,
							 private var l2RegParam: Double
						 ) extends Serializable {
	require(this.l2RegParam >= 0, s"Regularization parameter must be nonnegative but got ${this.l2RegParam}")
	val epsilon = 1e-8

	def setLearningRate(lr: Double): Updater = {
		this.learningRate = lr
		this
	}

	def setRegParam(regParam: Double): Updater = {
		require(this.l2RegParam >= 0,
			s"Regularization parameter must be nonnegative but got ${this.l2RegParam}")
		this.l2RegParam = regParam
		this
	}

	def compute(w: Vector, g: Vector, i: Int): (Vector, Double)
}


/**
	* :: DeveloperApi ::
	* A simple updater for gradient descent with L2 regularization.
	*/
@DeveloperApi
class SimpleUpdater(
										 private var learningRate: Double,
										 private var l2RegParam: Double
									 ) extends Updater(learningRate, l2RegParam) {
	def this(lr: Double) = this(lr, 0.0)

	override def compute(
							 weightsOld: Vector,
							 gradients: Vector,
							 iter: Int
						 ): (Vector, Double) = {
		val actualGradients = toBreeze(gradients) + this.l2RegParam * toBreeze(weightsOld)
		val weights = toBreeze(weightsOld) - this.learningRate * actualGradients
		val regLoss = if (this.l2RegParam == 0) 0 else 0.5 * this.l2RegParam * (toBreeze(weightsOld) :* toBreeze(weightsOld)).sum
		(fromBreeze(weights), regLoss)
	}
}

/**
	* :: DeveloperApi ::
	* A simple momentum updater for gradient descent with l2 regularization.
	* v = v * gamma + grad * lr
	* w = w - v
	*/
@DeveloperApi
class MomentumUpdater(
											 private var learningRate: Double,
											 private var gamma: Double,
											 private var l2RegParam: Double
										 ) extends Updater(learningRate, l2RegParam) {
	private var mementum: Vector = _
	def this(lr: Double, gamma: Double) = this(lr, gamma, 0.0)
	def this(lr: Double) = this(lr, 0.9, 0.0)
	def this() = this(0.01)

	def setGamma(gamma: Double): MomentumUpdater = {
		this.gamma = gamma
		this
	}

	def initializeMomentum(N: Int): MomentumUpdater = {
		this.mementum = Vectors.zeros(N)
		this
	}

	override def compute(
							 weightsOld: Vector,
							 gradients: Vector,
							 iter: Int
						 ): (Vector, Double) = {
		val actualGradients = toBreeze(gradients) + this.l2RegParam * toBreeze(weightsOld)
		val accum = this.gamma * toBreeze(this.mementum) + this.learningRate * actualGradients
		val weights = toBreeze(weightsOld) - accum
		this.mementum = fromBreeze(accum)
		val regLoss = if (this.l2RegParam == 0) 0 else 0.5 * this.l2RegParam * (toBreeze(weightsOld) :* toBreeze(weightsOld)).sum
		(fromBreeze(weights), regLoss)
	}
}


/**
	* :: DeveloperApi ::
	* Adagrad updater for gradient descent with l2 regularization.
	* s = s + grad * grad
	* w = w - lr * grad / sqrt(s + eps)
	* @param learningRate
	* @param l2RegParam
	*/
@DeveloperApi
class AdagradUpdater(
											private var learningRate: Double,
											private var l2RegParam: Double
										) extends Updater(learningRate, l2RegParam) {
	private var square: Vector = _
	def this(lr: Double) = this(0.1, 0.0)
	def this() = this(0.1)

	def initializeSquare(N: Int): AdagradUpdater = {
		this.square = Vectors.zeros(N)
		this
	}

	override def compute(
												weightsOld: Vector,
											 	gradients: Vector,
											 	iter: Int
											): (Vector, Double) = {
		val actualGradients = toBreeze(gradients) + this.l2RegParam * toBreeze(weightsOld)
		val accum = toBreeze(this.square) + actualGradients :* actualGradients
		val sqrtHistGrad = accum.map(k => sqrt(k + this.epsilon))
		val weights = toBreeze(weightsOld) - this.learningRate * (actualGradients :/ sqrtHistGrad)
		this.square = fromBreeze(accum)
		val regLoss = if (this.l2RegParam == 0) 0 else 0.5 * this.l2RegParam * (toBreeze(weightsOld) :* toBreeze(weightsOld)).sum
		(fromBreeze(weights), regLoss)
	}
}

/**
	* :: DeveloperApi ::
	* RMSProp updater for stochastic gradient descent with l2 regularization.
	* s = gamma * s + (1-gamma) * g * g
	* w = w - lr * g / sqrt(s + eps)
	* @param learningRate
	* @param gamma
	* @param l2RegParam
	*/
@DeveloperApi
class RMSPropUpdater(
											private var learningRate: Double,
											private var gamma: Double,
											private var l2RegParam: Double
										) extends Updater(learningRate, l2RegParam) {
	private var square: Vector = _
	def this(lr: Double, gamma: Double) = this(lr, gamma, 0.0)
	def this(lr: Double) = this(lr, 0.999, 0.0)
	def this() = this(0.1)

	def setGamma(gamma: Double): RMSPropUpdater = {
		this.gamma = gamma
		this
	}

	def initializeSquare(N: Int): RMSPropUpdater = {
		this.square = Vectors.zeros(N)
		this
	}

	override def compute(
												weightsOld: Vector,
												gradients: Vector,
												iter: Int
											): (Vector, Double) = {
		val actualGradients = toBreeze(gradients) + this.l2RegParam * toBreeze(weightsOld)
		val accum = this.gamma * toBreeze(this.square) + (1-this.gamma) * (actualGradients :* actualGradients)
		val sqrtHistGrad = accum.map(k => sqrt(k + this.epsilon))
		val weights = toBreeze(weightsOld) - this.learningRate * (actualGradients :/ sqrtHistGrad)
		this.square = fromBreeze(accum)
		val regLoss = if (this.l2RegParam == 0) 0 else 0.5 * this.l2RegParam * (toBreeze(weightsOld) :* toBreeze(weightsOld)).sum
		(fromBreeze(weights), regLoss)
	}
}

/**
	* :: DeveloperApi ::
	* t = t + 1
	* v = beta1 * v + (1 - beta1) * g
	* s = beta2 * s + (1 - beta2) * g :* g
	* v_ = v / (1 - beta1^t)
	* s_ = s / (1 - beta2^t)
	* w = w - lr * v_ / sqrt(s_ + eps)
	* @param learningRate
	* @param beta1
	* @param beta2
	* @param l2RegParam
	*/
@DeveloperApi
class AdamUpdater(
									 private var learningRate: Double,
									 private var beta1: Double=0.9,
									 private var beta2: Double=0.999,
									 private var l2RegParam: Double=0.0
								 ) extends Updater(learningRate, l2RegParam) {
	private var momentum: BV[Double] = _
	private var square: BV[Double] = _
	private var beta1Power = this.beta1
	private var beta2Power = this.beta2
	def this(lr: Double, beta1: Double, beta2: Double) = this(lr, beta1, beta2, 0.0)
	def this() = this(0.1)

	def setBeta1(b1: Double): AdamUpdater = {
		require(b1 >= 0 && b1 < 1.0, s"beta1 must greater than or equals 0 and smaller than 1.0, but got $b1")
		this.beta1 = b1
		this
	}

	def setBeta2(b2: Double): AdamUpdater = {
		require(b2 >= 0 && b2 < 1.0, s"beta1 must greater than or equals 0 and smaller than 1.0, but got $b2")
		this.beta2 = b2
		this
	}

	def initialMomentum(N: Int): AdamUpdater = {
		this.momentum = BDV.zeros[Double](N)
		this
	}

	def initialSquare(N: Int): AdamUpdater = {
		this.square = BDV.zeros[Double](N)
		this
	}

	override def compute(
												weightsOld: Vector,
												gradients: Vector,
												iter: Int
											): (Vector, Double) = {
		val actualGradients = toBreeze(gradients) + this.l2RegParam * toBreeze(weightsOld)
		val momentum_t = this.beta1 * this.momentum + (1 - this.beta2) * actualGradients
		val square_t = this.beta2 * this.square + (1 - this.beta2) * (actualGradients :* actualGradients)
		val momentum_t_ = momentum_t / (1 - this.beta1Power)
		val square_t_ = square_t / (1 - this.beta2Power)
		val sqrtHistGrad = square_t_.map(k => sqrt(k + this.epsilon))
		val weights = toBreeze(weightsOld) - this.learningRate * (momentum_t_ :/ sqrtHistGrad)
		val regLoss = if (this.l2RegParam == 0) 0 else 0.5 * this.l2RegParam * (toBreeze(weightsOld) :* toBreeze(weightsOld)).sum
		this.square = square_t
		this.momentum = momentum_t
		this.beta1Power *= this.beta1
		this.beta2Power *= this.beta2
		(fromBreeze(weights), regLoss)
	}
}


