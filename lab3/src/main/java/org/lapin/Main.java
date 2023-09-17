package org.lapin;
import org.jpl7.*;
import org.lapin.parser.Parser;

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {

        System.out.println("Hello and welcome!");
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter your query: ");
        String query = scanner.nextLine();
        Query q1 =
                new Query(
                        "consult",
                        new Term[] {new Atom("/Users/aleksei/ITMO/СИИ-2023/new-lab-1/main.pl")}
                );
        if (q1.hasSolution()) {

            String res = Parser.run(query);
            Query q3 = new Query(res);
            var solutions = q3.allSolutions();
            System.out.println("Вам подойдут игры: ");
            for (int i = 0; i < solutions.length; i++) {
                System.out.println(solutions[i].get("X"));
            }
        } else {
            System.out.println("Connect to Prolog Failed");
        }
    }
}