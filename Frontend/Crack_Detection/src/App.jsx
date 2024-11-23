import { useState } from 'react'
import './App.css'
import image1 from './testPictures/image_crack_neg_1.jpg'
import image2 from './testPictures/image_crack_pos_1.jpg'
import Image from './components/Image'

function App() {

  const [selectedImage, setSelectedImage] = useState(null);
  const list = [image1, image2];
  const [evaluationResult, setEvaluationResult] = useState(null);


  const handleImageSelect = (image) => {
    setSelectedImage(image);
    console.log("Image selected in parent:", image);};


  const evaluateModel = async () => {
    const response = await fetch('http://localhost:5000/evaluate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: selectedImage,  // You would pass your actual model object here
        test_loader: {},  // Replace with your actual test data
        device: 'cpu',
      }),
    });

    if (response.ok) {
      const data = await response.json();
      setEvaluationResult(data);
      console.log('Evaluation Result:', data);
    } else {
      console.error('Error evaluating the model');
    }
  };
  
  

  return (
    <> 
    <div class="header"> 
      <h1>Crack Detection</h1>
    </div>
    <center>
    <div class="body-content">
      <div>
        <Image list={list} onSelect={handleImageSelect} />
        {selectedImage && selectedImage !== "Please choose one option" && (
          <div className="selected-image">
  <img src={selectedImage} alt="selectedImage" />
  <button onClick={evaluateModel} > Test selected picture </button>
          </div>
)}
        
      </div>

      <div className="parent">
        <div className="child inline-block-child">
        {evaluationResult && (
        <div>
          <h3>Evaluation Results</h3>
          <p>Accuracy: {evaluationResult.accuracy}</p>
          <p>Precision: {evaluationResult.precision}</p>
          <p>Recall: {evaluationResult.recall}</p>
          <p>F1 Score: {evaluationResult.f1}</p>
        </div>
      )}
          


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
