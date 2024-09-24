First, We need to run `disease prediction.ipynb` and `dna reconstruction.ipynb` file using Jupyter notebook. Ensure that `python` and other necessary libraries like `flask`, `pandas`, `numpy`, `joblib`, `scikit-learn` are properly installed. Install any missing libraries using `PIP` package manager.


Dump these files using these command from the mentioned files.
```python
# 'disease prediction.ipynb'
joblib.dump(model, 'dna_disease_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

# 'dna reconstruction.ipynb'
joblib.dump(model, 'dna_reconstruction_model.pkl')
```

Finally, run the `app.py` file using following command:
```bash
python app.py
```

By default, this will run the model at `http://127.0.0.1:5000/`
