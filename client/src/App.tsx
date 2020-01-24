import React, { useEffect } from "react";
import "./App.css";

const App: React.FC = () => {
	useEffect(() => {
		fetch("/emotions").then(response => {
			response.json().then(data => {
				console.log(data);
			});
		});
	}, []);

	return <div className='App'></div>;
};

export default App;
