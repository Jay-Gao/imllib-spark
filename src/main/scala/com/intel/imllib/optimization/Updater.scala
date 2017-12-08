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

import org.apache.spark.annotation.DeveloperApi
import breeze.linalg.{DenseVector => BDV, Vector=>BV}
import breeze.numerics.sqrt

import scala.collection.mutable.ArrayBuffer


abstract class Updater(
							 private var learningRate: Double,
							 private var l2RegParam: Double
						 ) extends Serializable {
	require(this.l2RegParam >= 0, s"Regularization parameter must be nonnegative but got ${this.l2RegParam}")
	val epsilon = 1e-8

	def setLearningRate(lr: Double): this.type = {
		this.learningRate = lr
		this
	}

	def setRegParam(regParam: Double): this.type = {
		require(this.l2RegParam >= 0,
			s"Regularization parameter must be nonnegative but got ${this.l2RegParam}")
		this.l2RegParam = regParam
		this
	}

	def initialize(sizes: Int*): this.type

	def compute(w: Array[BV[Double]], g: Array[BV[Double]], i: Int): (Array[BV[Double]], Double)
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

	override def initialize(sizes: Int*): SimpleUpdater.this.type = this

	override def compute(
							 weightsOld: Array[BV[Double]],
							 gradients: Array[BV[Double]],
							 iter: Int
						 ): (Array[BV[Double]], Double) = {
		require(weightsOld.length == gradients.length)
		val weights = new Array[BV[Double]](weightsOld.length)
		var regLoss = 0.0
		for (i <- weightsOld.indices) {
			val actualGradients = gradients(i) + this.l2RegParam * weightsOld(i)
			weights(i) = weightsOld(i) - this.learningRate * actualGradients
			regLoss += (if (this.l2RegParam == 0) 0 else 0.5 * this.l2RegParam * weightsOld(i).dot(weightsOld(i)))
		}
		(weights, regLoss)
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
	private val momentums = new ArrayBuffer[BV[Double]]()
	def this(lr: Double, gamma: Double) = this(lr, gamma, 0.0)
	def this(lr: Double) = this(lr, 0.9, 0.0)
	def this() = this(0.01)

	def setGamma(gamma: Double): this.type = {
		this.gamma = gamma
		this
	}

	def initialize(sizes: Int*): this.type = {
		sizes.foreach {
			s =>
				val momentum = BV.zeros[Double](s)
				this.momentums.append(momentum)
		}
		this
	}

	override def compute(
							 weightsOld: Array[BV[Double]],
							 gradients: Array[BV[Double]],
							 iter: Int
						 ): (Array[BV[Double]], Double) = {
		require(weightsOld.length == gradients.length)
		val weights = new Array[BV[Double]](weightsOld.length)
		var regLoss = 0.0
		for (i <- weightsOld.indices) {
			val actualGradients = gradients(i) + this.l2RegParam * weightsOld(i)
			this.momentums(i) = this.gamma * this.momentums(i) + this.learningRate * actualGradients
			weights(i) = weightsOld(i) - this.momentums(i)
			regLoss += (if (this.l2RegParam == 0) 0 else 0.5 * this.l2RegParam * weightsOld(i).dot(weightsOld(i)))
		}
		(weights, regLoss)
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
	private val squares = new ArrayBuffer[BV[Double]]()
	def this(lr: Double) = this(0.1, 0.0)
	def this() = this(0.1)

	def initialize(sizes: Int*): this.type = {
		sizes.foreach {
			s =>
				val square = BV.zeros[Double](s)
				this.squares.append(square)
		}
		this
	}

	override def compute(
												weightsOld: Array[BV[Double]],
												gradients: Array[BV[Double]],
												iter: Int
											): (Array[BV[Double]], Double) = {
		require(weightsOld.length == gradients.length)
		val weights = new Array[BV[Double]](weightsOld.length)
		var regLoss = 0.0
		for (i <- weightsOld.indices) {
			val actualGradients = gradients(i) + this.l2RegParam * weightsOld(i)
			this.squares(i) = this.squares(i) + actualGradients * actualGradients
			weights(i) = weightsOld(i) - this.learningRate * (actualGradients / sqrt(this.squares(i) + this.epsilon))
			regLoss += (if (this.l2RegParam == 0) 0 else 0.5 * this.l2RegParam * weightsOld(i).dot(weightsOld(i)))
		}
		(weights, regLoss)
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
	private val squares = new ArrayBuffer[BV[Double]]()
	def this(lr: Double, gamma: Double) = this(lr, gamma, 0.0)
	def this(lr: Double) = this(lr, 0.999, 0.0)
	def this() = this(0.1)

	def setGamma(gamma: Double): RMSPropUpdater = {
		this.gamma = gamma
		this
	}

	def initialize(sizes: Int*): this.type = {
		sizes.foreach {
			s =>
				val square = BV.zeros[Double](s)
				this.squares.append(square)
		}
		this
	}

	override def compute(
												weightsOld: Array[BV[Double]],
												gradients: Array[BV[Double]],
												iter: Int
											): (Array[BV[Double]], Double) = {
		require(weightsOld.length == gradients.length)
		val weights = new Array[BV[Double]](weightsOld.length)
		var regLoss = 0.0
		for (i <- weightsOld.indices) {
			val actualGradients = gradients(i) + this.l2RegParam * weightsOld(i)
			this.squares(i) = this.gamma * this.squares(i) + (1 - this.gamma) * (actualGradients * actualGradients)
			weights(i) = weightsOld(i) - this.learningRate * (actualGradients / sqrt(this.squares(i) + this.epsilon))
			regLoss += (if (this.l2RegParam == 0) 0 else 0.5 * this.l2RegParam * weightsOld(i).dot(weightsOld(i)))
		}
		(weights, regLoss)
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
	private val momentums = new ArrayBuffer[BV[Double]]()
	private val squares = new ArrayBuffer[BV[Double]]()
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

	def initialize(sizes: Int*): this.type = {
		sizes.foreach {
			s =>
				this.squares.append(BV.zeros[Double](s))
				this.momentums.append(BV.zeros[Double](s))
		}
		this
	}

	override def compute(
												weightsOld: Array[BV[Double]],
												gradients: Array[BV[Double]],
												iter: Int
											): (Array[BV[Double]], Double) = {
		require(weightsOld.length == gradients.length)
		val weights = new Array[BV[Double]](weightsOld.length)
		var regLoss = 0.0
		for (i <- weightsOld.indices) {
			val actualGradients = gradients(i) + this.l2RegParam * weightsOld(i)
			this.momentums(i) = this.beta1 * this.momentums(i) + (1 - this.beta2) * actualGradients
			this.squares(i) = this.beta2 * this.squares(i) + (1 - this.beta2) * (actualGradients * actualGradients)
			val momentum_t = this.momentums(i) / (1 - this.beta1Power)
			val square_t = this.squares(i) / (1 - this.beta2Power)
			weights(i) = weightsOld(i) - this.learningRate * (momentum_t / sqrt(square_t + this.epsilon))
			regLoss += (if (this.l2RegParam == 0) 0 else 0.5 * this.l2RegParam * weightsOld(i).dot(weightsOld(i)))

		}
		this.beta1Power *= this.beta1
		this.beta2Power *= this.beta2
		(weights, regLoss)
	}
}



