import { useState } from 'react'
import './App.css'
import image1 from './testPictures/image_crack_neg_1.jpg'
import image2 from './testPictures/image_crack_pos_1.jpg'
import Image from './components/Image'

function App() {

  const [selectedImage, setSelectedImage] = useState(null);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const list = [image1, image2];


  const handleTestImage = async () => {
    if (!selectedImage) {
      alert('Please select an image first.');
      return;
    }

    try {
      // Send POST request to /predict endpoint
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_path: selectedImage,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch prediction results');
      }

      // Parse the JSON response
      const data = await response.json();
      
      // Assuming the response contains evaluation results like accuracy, precision, etc.
      setEvaluationResults(data.predictions);
      console.log('Prediction results:', data.predictions);
    } catch (error) {
      console.error('Error testing image:', error);
      alert('Error during prediction. Please try again.');
    }
  };

  const handleImageSelect = (image) => {
    setSelectedImage(image);
    console.log("Image selected in parent:", image);};




  return (
    <> 
    <div class="header"> 
      <h1>Crack Detection</h1>
    </div>
    <center>
    <div className="body-content">
      <div>
        <Image list={list} onSelect={handleImageSelect} />
        {selectedImage && selectedImage !== "Please choose one option" && (
          <div className="selected-image">
  <img src={selectedImage} alt="selectedImage" />
  <button onClick={handleTestImage}> Test selected picture </button>
          </div>
)}
        
      </div>

      <div className="parent">
        <div className="child inline-block-child">
        
        <div>
          <h3>Evaluation Results</h3>
          <p>Accuracy: </p>
          <p>Precision: </p>
          <p>Recall: </p>
          <p>F1 Score: </p>
        </div>
      
          


        </div>
        <div className="child inline-block-child">2</div>
        <div className="child inline-block-child">3</div>
        <div className="child inline-block-child">4</div>
      </div>
    </div>
    </center>

    </>
  )
}

export default App