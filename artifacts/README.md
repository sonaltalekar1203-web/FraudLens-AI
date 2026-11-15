# Model artifacts

The trained FraudLens AI model and dashboard data are intentionally not committed to this repository by default.

The application expects the model artifact and metadata to be supplied separately in the locations configured by the API. For local setup, copy the approved files into this directory after downloading them from your private project storage.

Do not place raw PaySim data, private transaction records, MLflow databases, credentials, or API tokens here. For large model files, use Git LFS or a controlled artifact store and document the retrieval process.
