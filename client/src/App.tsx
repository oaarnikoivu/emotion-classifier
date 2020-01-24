import React, { useEffect, useState } from "react";
import "./App.css";
import { Emotions } from "./components/emotions_component";

const App: React.FC = () => {
	const [emotions, setEmotions] = useState([]);

	useEffect(() => {
		fetch("/emotions").then(response => {
			response.json().then(data => {
				setEmotions(data.Test);
			});
		});
	}, []);

	console.log(emotions);

	return (
		<div className='App'>
			<Emotions emotions={emotions} />
		</div>
	);
};

export default App;
