
# 📊 Web Log Analyzer with Machine Learning
This is a Flask-based web application designed to analyze web server log files using a combination of traditional data visualization and machine learning techniques. The application supports log file upload, data preprocessing, visual analytics, classification, clustering, anomaly detection, and downloadable PDF reports.

🚀 Features

✅ Upload and Analyze Logs
Upload .csv log files for processing and analysis.

Expected CSV columns: IP, Request, StatusCode, Column8, UserAgent, ResponseTime.

📈 Data Visualizations
Bar Chart: Request type distribution.

Heatmap: Association rules between request types and response time bins.

K-Means Clustering: Grouping based on StatusCode and ResponseTime.

🧠 Machine Learning Models
Decision Tree Classifier

K-Nearest Neighbors (KNN)

Isolation Forest (for anomaly detection)

Each model generates:

Accuracy metrics

Classification reports

Confusion matrices

Visual insights (PNG charts)

📄 PDF Reporting
Generates a downloadable professional PDF report of all analysis results.

🖥️ Technologies Used
Backend: Flask

Data Handling: Pandas, NumPy

ML Models: Scikit-learn, MLxtend (Apriori, Association Rules)

Visualization: Matplotlib, Seaborn

PDF Generation: xhtml2pdf, ReportLab

📂 Project Structure
project/
│
├── static/                     # Contains generated image files (charts, heatmaps)
├── templates/
│   └── index.html              # Upload page
│   └── results.html            # Results display page
├── uploads/                    # Stores uploaded CSV log files
├── app.py                      # Main Flask application
└── README.md                   # You're here!
✅ How to Run

Step 1: Clone the Repository
git clone https://github.com/YashM-235/LOG-LENS.git

cd log-analyzer-flask

Step 2: Install Dependencies    pip install -r requirements.txt

Step 3: Run the App     python app.py

📥 Upload File Format
Ensure your CSV file has the following format (headers optional but will be overwritten internally):

IP, Request, StatusCode, Column8, UserAgent, ResponseTime
📤 Sample Output
📊 static/bar_chart.png: Request type distribution

🔥 static/heatmap.png: Association rule heatmap

🎯 static/kmeans_clusters.png: KMeans clustering plot

🧠 static/decision_tree_classification_report.png: Decision Tree metrics

🕵️‍♀️ static/isolation_forest.png: Anomaly detection result

📌 Future Enhancements
📁 Multi-file Upload Support
Currently, only one log file is processed at a time.

Future versions will support batch processing of multiple log files and provide aggregated reports.

🧾 Enhanced PDF Reporting
Add more dynamic visualizations and summaries.

Include interactive charts (using Plotly or Altair).

📊 Additional ML Models
Random Forest, Gradient Boosting, SVM, or Neural Networks for better accuracy.

AutoML pipeline integration.

🌐 API Endpoints
Provide RESTful API for automated log analysis and integration with other tools.

🕵️‍♂️ Security Improvements
Implement file validation, file size limits, and secure filename handling.

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change or add.

📜 License
This project is licensed under the MIT License.

📧 Contact

Developed by Yash Mehta & Priyanshu

Email: yash.dlw@gmail.com
GitHub: [@YashM-235]
