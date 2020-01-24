import React, { useEffect, useState } from "react";
import "./App.css";
import { Emotions } from "./components/emotions_component";
import { EmotionForm } from "./components/emotion_form_component";
import { Container } from "semantic-ui-react";

const App: React.FC = () => {
	const [emotion, setEmotion] = useState("");

	useEffect(() => {
		fetch("/emotions").then(response => {
			response.json().then(data => {
				setEmotion(data.prediction);
			});
		});
	}, []);

	return (
		<Container style={{ marginTop: 40 }}>
			<EmotionForm
				onNewText={(text: string) => {
					setEmotion(text);
				}}
			/>
			<Emotions emotion={emotion} />
		</Container>
	);
};

export default App;
