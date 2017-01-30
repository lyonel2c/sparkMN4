#!/bin/bash
EXAMPLE_MASTER=${MASTER:-"local[*]"}

sh "${SPARK_HOME}"/bin/spark-submit  \
--driver-memory 5G \
--class org.deeplearning4j.deepMN4.foodRecognition.DPFSpark \
--executor-memory 5G \
--master $EXAMPLE_MASTER \
target/deepMN4-1.0-SNAPSHOT.jar

