document.addEventListener('DOMContentLoaded', function() {
    const diabetesForm = document.getElementById('diabetesForm');
    if (diabetesForm) {
        diabetesForm.addEventListener('submit', function(e) {
            e.preventDefault();

            const submitBtn = diabetesForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.textContent;
            submitBtn.textContent = 'Processing...';
            submitBtn.disabled = true;

            document.getElementById('result').style.display = 'none';
            document.getElementById('error').style.display = 'none';

            fetch('/predict_diabetes', {
                method: 'POST',
                body: new FormData(diabetesForm)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('predictionText').textContent = 
                        `Prediction: ${data.result} (Confidence: ${data.confidence}%)`;
                    document.getElementById('result').style.display = 'block';
                } else {
                    document.getElementById('errorText').textContent = data.error;
                    document.getElementById('error').style.display = 'block';
                }
            })
            .catch(error => {
                document.getElementById('errorText').textContent = 'An error occurred while processing your request.';
                document.getElementById('error').style.display = 'block';
            })
            .finally(() => {
                submitBtn.textContent = originalText;
                submitBtn.disabled = false;
            });
        });
    }

    const heartForm = document.getElementById('heartForm');
    if (heartForm) {
        heartForm.addEventListener('submit', function(e) {
            e.preventDefault();

            const submitBtn = heartForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.textContent;
            submitBtn.textContent = 'Processing...';
            submitBtn.disabled = true;

            document.getElementById('result').style.display = 'none';
            document.getElementById('error').style.display = 'none';

            fetch('/predict_heart', {
                method: 'POST',
                body: new FormData(heartForm)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('predictionText').textContent = 
                        `Prediction: ${data.result} (Confidence: ${data.confidence}%)`;
                    document.getElementById('result').style.display = 'block';
                } else {
                    document.getElementById('errorText').textContent = data.error;
                    document.getElementById('error').style.display = 'block';
                }
            })
            .catch(error => {
                document.getElementById('errorText').textContent = 'An error occurred while processing your request.';
                document.getElementById('error').style.display = 'block';
            })
            .finally(() => {
                submitBtn.textContent = originalText;
                submitBtn.disabled = false;
            });
        });
    }

    const skinCancerForm = document.getElementById('skinCancerForm');
    if (skinCancerForm) {
        const fileInput = document.getElementById('file');
        const imagePreview = document.getElementById('imagePreview');
        const previewImg = document.getElementById('preview');
        
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                
                reader.addEventListener('load', function() {
                    previewImg.setAttribute('src', this.result);
                    imagePreview.style.display = 'block';
                });
                
                reader.readAsDataURL(file);
            } else {
                imagePreview.style.display = 'none';
            }
        });
        
        skinCancerForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const submitBtn = skinCancerForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.textContent;
            submitBtn.textContent = 'Analyzing...';
            submitBtn.disabled = true;
            document.getElementById('result').style.display = 'none';
            document.getElementById('error').style.display = 'none';

            fetch('/predict_skin_cancer', {
                method: 'POST',
                body: new FormData(skinCancerForm)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('predictionText').textContent = 
                        `Prediction: ${data.result} (Confidence: ${data.confidence}%)`;
                    document.getElementById('result').style.display = 'block';
                } else {
                    document.getElementById('errorText').textContent = data.error;
                    document.getElementById('error').style.display = 'block';
                }
            })
            .catch(error => {
                document.getElementById('errorText').textContent = 'An error occurred while processing your request.';
                document.getElementById('error').style.display = 'block';
            })
            .finally(() => {
                submitBtn.textContent = originalText;
                submitBtn.disabled = false;
            });
        });
    }
});