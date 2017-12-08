#!/usr/bin/env bash
$SPARK_HOME/bin/spark-submit \
    --class FMExample \
    --master local[*] \
    target/scala-2.11/imllib_2.11-0.0.1.jar \
    file:///Users/jay/Projects/open_sources/imllib-spark/data/fm/a9a \
    2 \
    40 \
    0.01 \
    2 \
    bin/template/FMModel
