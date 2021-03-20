# Title     : detectChangePointwithCPM.R
# Objective : Simple R file to call the cpm library
# Created by: Harold
# Created on: 1/26/2021

library(cpm)

Detect_r_cpm_GaussianChangePoint = function(batch_waiting_times) {
  # print(typeof(batch_waiting_times))
  #resultsStudent <- detectChangePointBatch(batch_waiting_times, "GLRAdjusted", 0.05)
  resultsStudent <- detectChangePoint(batch_waiting_times, cpmType = "GLR", ARL0 = 50000, startup = 20)
  #print(resultsStudent)
  if (resultsStudent$changeDetected) {
    output_val <- resultsStudent$detectionTime
    #    output_val <- resultsStudent$changePoint # used that with Batch detection
  }else {
    output_val <- -1
  }
  return(output_val)
}


Detect_r_cpm_NonParametricChangePoint = function(waiting_times) {
  results <- detectChangePoint(waiting_times, cpmType = "Cramer-von-Mises", ARL0 = 50000, startup = 20)
  if (results$changeDetected) {
    output_val <- results$detectionTime
  }else {
    output_val <- -1
  }
  return(output_val)
}