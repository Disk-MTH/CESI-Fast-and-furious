import os
import mysql.connector
import matplotlib.pyplot as plt
import xml.etree.ElementTree as XmlParser


def load_data_to_db(db, folder_path):
    cursor = db.cursor()  # create a cursor to execute queries

    cursor.execute("DROP DATABASE IF EXISTS meca_project_tracking;")  # drop the database if it exists
    cursor.execute("CREATE DATABASE meca_project_tracking;")  # create the database
    cursor.execute("USE meca_project_tracking;")  # select the database

    counter = 1
    for folders, sub_folders, files in os.walk(folder_path):    # walk through all the files in the folder
        for file in files:  # for each file
            if ".Trk" in file.title() and counter:  # if the file is a tracking file
                cursor.execute(
                    "CREATE TABLE tracking_" +
                    str(counter) +
                    " (time double NOT NULL, x_pos double NOT NULL, y_pos double NOT NULL);"
                )  # create a table for the tracking file

                xml_tree_root = XmlParser.parse(os.path.join(folders, file)).getroot()  # load file in parser

                t, x, y = [0], [], []  # create lists for time, x and y positions

                for x_pos in xml_tree_root.findall(".//property[@name='x']"):  # get all the x positions
                    x.append(float(x_pos.text))

                for y_pos in xml_tree_root.findall(".//property[@name='y']"):  # get all the y positions
                    y.append(float(y_pos.text))

                for i in range(len(x)):  # create the time list
                    t.append(t[i] + float(xml_tree_root.find(".//property[@name='delta_t']").text) / 1000)

                x_max = max(x)  # get the max x position
                for i in range(len(x)):  # for each x position
                    x[i] = x_max - x[i]  # invert the x position

                y_max = max(y)  # get the max y position
                for i in range(len(y)):  # for each y position
                    y[i] = y_max - y[i]  # invert the y position

                cursor.execute(("INSERT INTO tracking_" + str(counter) + " (time, x_pos, y_pos) VALUES " +
                                str([(ti, xi, yi) for ti, xi, yi in zip(t, x, y)]) + ";")
                               .replace("[", "").replace("]", "")
                               )  # insert the data into the table

                counter += 1


def plot_tracking(db, index):
    cursor = db.cursor()  # create a cursor to execute queries

    cursor.execute("USE meca_project_tracking;")  # select the database
    cursor.execute("SELECT * FROM tracking_" + str(index) + ";")  # select the data from the table

    t, x, y = zip(*cursor.fetchall())

    path = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    x_pos = plt.subplot2grid((2, 2), (1, 0))
    y_pos = plt.subplot2grid((2, 2), (1, 1))

    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    path.plot(x, y, label="Path (mm)")
    path.set_title("Path of the object")
    path.legend()

    x_pos.plot(t, x, label="X position (mm/s)")
    x_pos.set_title("X position as a function of time")
    x_pos.legend()

    y_pos.plot(t, y, label="Y position (mm/s)")
    y_pos.set_title("Y position as a function of time")
    y_pos.legend()

    plt.title("Tracking " + str(index))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    database = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database=""
    )

    load_data_to_db(database, os.getcwd() + "/tracking")
    plot_tracking(database, 1)

    database.close()
