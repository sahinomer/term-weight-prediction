# Term Weight Prediction Model

BERT based regression model that takes term and the query as input, and the model estimates target term weight which can be term recall or pairwise optimized ones. The proposed BERT regression model allows to phrase weighting like term weighting as distinct from the term estimation frameworks in the literature thanks to the term-query input design.

![BERT Query Term Weighting Model Architecture](./bert-termweight.png)


## Example

Train term weight prediction model
```
python bert_qtw.py \
    --train_queries term-weight.train.json \
    --dev_queries term-recall.dev.small.json \
    --output predictions.json
```

#### train.json
```json
...
{
    "qid": "121352", 
    "query": "define extreme", 
    "term_weight": 
        {
            "define": 0.10255750445487254, 
            "extreme": 0.9863207054680766
        }
}
...
```

#### dev.small.json
```json
...
{
    "qid": "2", 
    "query": " Androgen receptor define", 
    "term_recall": 
        {
            "androgen": 1.0, 
            "receptor": 1.0, 
            "define": 0
        }
}
...
```

## Paper

- Ömer Şahin, İlyas Çiçekli and Gönenç Ercan, "Learning Term Weights by Overfitting Pairwise Ranking Loss" *Turkish Journal of Electrical Engineering and Computer Sciences,* doi:10.3906/elk-2112-13 [Link](https://aj.tubitak.gov.tr/elektrik/issues/elk-22-30-5/elk-30-5-16-2112-13.pdf)

- Ömer Şahin, "EVALUATING THE USE OF NEURAL RANKING METHODS IN SEARCH ENGINES" *Graduate School of Science and Engineering of Hacettepe University, Degree of Master of Science In Computer Engineering* [Link](https://tez.yok.gov.tr/UlusalTezMerkezi/TezGoster?key=CG8WvdvvxJP04Unr7Yecf2S6VP-tW7orh9h_lec-PrtWf2J8Wg5BbHXZkkIw1uj4)