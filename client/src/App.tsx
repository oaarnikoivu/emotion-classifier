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
		<>
			<Container style={{ marginTop: 40 }}>
				<div style={{ paddingBottom: 10 }}>
					{
						"This AI has been trained to understand emotion from Tweets. Type a sentence to see what the AI algorithm thinks."
					}
				</div>
				<EmotionForm
					onNewText={(text: string) => {
						setEmotion(text);
					}}
				/>
				<Emotions emotion={emotion} />
			</Container>
		</>
	);
};

export default App;
