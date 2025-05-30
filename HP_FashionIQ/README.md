# Human-Prefernce FashionIQ (HP-FashionIQ) dataset

The commonly used evaluation metric, **Recall@k**, fails to fully capture user satisfaction. While user satisfaction generally increases as more relevant items are retrieved, Recall@k only verifies whether a specific target image appears—ignoring the relevance of other retrieved results.

Evaluating image relevance in **Composed Image Retrieval (CIR)** is especially challenging, as it requires assessing how well each image aligns with both textual and visual inputs. This often involves subtle and complex attribute matching. Human evaluation remains the most reliable method for this task, as humans can accurately judge how well an image satisfies a multi-modal query.

To support this evaluation, we introduce the **Human-Preference FashionIQ (HP-FashionIQ)** dataset, built from the validation set of the [FashionIQ dataset](https://github.com/XiaoxiaoGuo/fashion-iq), with human annotations from 61 participants. FashionIQ was selected for its detailed attribute supervision and alignment with real-world e-commerce search behaviors.

For more information, see **Section 4** and **Appendix C** of our paper.

## Preparation

1. Download and set up the [FashionIQ dataset](https://github.com/XiaoxiaoGuo/fashion-iq).
2. Replace all instances of `"-"` in the `hpfiq.json` file with the absolute path to your FashionIQ dataset directory.
3. Modify the `load_model` and `get_score_of_image` functions in `evaluate.py` to use your own model and scoring method.

## Preference Ratio

To evaluate how well a CIR model aligns with human preferences, we define the following metric:

$$\mathbb{P}(A \succ B \mid s_{\text{rel}}(A) > s_{\text{rel}}(B))$$

This represents the probability that set **A** is preferred over set **B** by humans, given that the model assigns a higher relevance score to **A** than to **B**.


## Run

To compute the preference ratio for a CIR model:

```bash
python -m evaluate --config_path=../configs/fashionIQ/eval.json
```

This will evaluate how closely your model’s relevance scores align with human preferences in the HP-FashionIQ dataset.