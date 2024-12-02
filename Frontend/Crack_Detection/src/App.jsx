import { useState } from 'react'
import './App.css'
import image1 from './testPictures/image_crack_neg_1.jpg'
import image2 from './testPictures/image_crack_pos_1.jpg'
import image3 from './testPictures/image_crack_pos_2.jpg'
import image4 from './testPictures/image_crack_neg_2.jpg'
import image5 from './testPictures/image_crack_neg_3.jpg'
import image6 from './testPictures/image_crack_pos_3.jpg'
import image7 from './testPictures/image_crack_pos_4.jpg'
import Image from './components/Image'

function App() {

  const [selectedImage, setSelectedImage] = useState(null);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const list = [image1, image2, image3, image4, image5, image6, image7];


  const handleTestImage = async () => {
    if (!selectedImage) {
      alert('Please select an image first.');
      return;
    }

    try {
      // Send POST request to /predict endpoint
      const response = await fetch('http://127.0.0.1:5000/predict', {
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
        <div className="child inline-block-child">Model 1: <div>{evaluationResults?.Model_1}</div> </div>
        <div className="child inline-block-child">Model 2: <div>{evaluationResults?.Model_2}</div> </div>
        <div className="child inline-block-child">Model 3: <div>{evaluationResults?.Model_3}</div> </div>
        <div className="child inline-block-child">Trained_RESNET: <div>{evaluationResults?.Trained_RESNET}</div> </div>
      </div>
    </div>
    </center>

    </>
  )
}

export default App