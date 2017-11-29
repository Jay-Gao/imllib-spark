#!/usr/bin/env bash
$SPARK_HOME/bin/spark-submit \
    --class LRWithAdaExample \
    --master local[*] \
    target/scala-2.11/imllib_2.11-0.0.1.jar \
    file:///Users/jay/Projects/open_sources/imllib-spark/data/lr/a9a \
    file:///Users/jay/Projects/open_sources/imllib-spark/data/lr/a9a.t \
    4
