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

package com.intel.imllib

import breeze.linalg.norm
import org.apache.spark.mllib.linalg.Vector
import breeze.linalg.{DenseVector=>BDV}

package object util {
	def isConverged(previousWeights: Vector,
													currentWeights: Vector,
													convergenceTol: Double): Boolean = {
		// To compare with convergence tolerance.
		val previousBDV = new BDV[Double](previousWeights.toDense.values)
		val currentBDV = new BDV[Double](currentWeights.toDense.values)

		val a = previousWeights.toDense
		// This represents the difference of updated weights in the iteration.
		val solutionVecDiff: Double = norm(previousBDV - currentBDV)

		solutionVecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
	}
}
