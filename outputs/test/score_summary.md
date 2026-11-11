# AlignScore Summary per candidate number and baseline


 ```
               mean       std       min       max
candidate                                        
1          0.940243  0.171705  0.001858  0.998333
2          0.928733  0.187364  0.001705  0.998626
3          0.914079  0.187590  0.002783  0.998483
4          0.896945  0.215601  0.009410  0.998849
5          0.877579  0.227789  0.001236  0.998622
Baseline: 
mean    0.918327
std     0.196847
min     0.002374
max     0.998461
Name: alignscore, dtype: float64

Overall candidate mean AlignScore: 0.9115157696626687
Mean selected candidate AlignScore: 0.991044578610397
Overall baseline mean AlignScore: 0.9183274620314833

 ```

# SummaC and Rouge-L Scores
```
Evaluation results
------------------
Articles: 615
Mean Candidate SummaC: 0.8753065708933807
Mean Candidate ROUGE-L: 0.36584155511093186
Mean Baseline SummaC: 0.761174256480261
Mean Baseline ROUGE-L: 0.36751780812135637

Descriptive statistics:

Sc
 elected Candidates     candidate_summac  candidate_rougeL
count        615.000000        615.000000
mean           0.875307          0.365842
std            0.279645          0.185087
min            0.015757          0.062500
25%            0.983430          0.243163
50%            0.990957          0.315789
75%            0.993160          0.421053
max            0.996741          1.000000


B       baseline_summac  baseline_rougeL
caselineount       615.000000       615.000000
mean          0.761174         0.367518
std           0.354050         0.187188
min           0.008886         0.086957
25%           0.534101         0.244898
50%           0.989589         0.315789
75%           0.992850         0.420204
max           0.995997         1.000000

```