import React, { useState } from "react";

//take a list of options
const Image = ({ list, onSelect }) => {
    const [data, setData] = useState(undefined);

    const onOptionChangeHandler = (event) => {

        setData(event.target.value);
        console.log(
            "User Selected Value - ",
            event.target.value);
        onSelect(event.target.value);
    };
    return (
        <center>
            <h3>Select a picture to test</h3>

            <select onChange={onOptionChangeHandler}>
                <option>Please choose one option</option>
                {list.map((option, index) => {
                    return (
                        <option key={index} value={option}>
                            {option}
                        </option>
                    );
                })}
            </select>
            <h3> {data} </h3>
        </center>
    );
};

export default Image;