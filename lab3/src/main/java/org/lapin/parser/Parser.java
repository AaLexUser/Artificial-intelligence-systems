package org.lapin.parser;

public class Parser {
    public static String run(String str){
        String queryResult = "";
        // Splitting the input string on minus sign. Assumes the string contains multiple queries
        var queries = str.split("\s*-\s*");
        for(int queryIndex = 0; queryIndex < queries.length; queryIndex++){
            if(queries[queryIndex].contains("персонаж")){
                queries[queryIndex] = queries[queryIndex].substring(queries[queryIndex].indexOf(":")+1).trim();
                // split the query into smaller components based on 'и'
                var components = queries[queryIndex].split("\s*и\s*");
                queryResult += "(";
                for (int componentIndex = 0; componentIndex < components.length; componentIndex++){
                    queryResult += "(";
                    // split the components based on ','
                    var items = components[componentIndex].split(",\s*");
                    for (int itemIndex = 0; itemIndex < items.length; itemIndex++) {
                        String item = "'" + items[itemIndex].trim() + "'";
                        item = item.replaceAll("\\s*\\.\\s*", "");
                        queryResult += "appears_in_game(" + item + ", X)";
                        if(itemIndex != items.length-1){
                            // add semicolon for multiple items in the same component
                            queryResult += "; ";
                        }
                    }
                    queryResult += ") ";
                    if(componentIndex != components.length-1){
                        // add comma for multiple components
                        queryResult += ", ";
                    }
                }
                queryResult += ") ";
            }
            if(queries[queryIndex].contains("приставка")){
                queries[queryIndex] = queries[queryIndex].substring(queries[queryIndex].indexOf(":")+1).trim();
                var components = queries[queryIndex].split("\s*и\s*");
                queryResult += "(";
                for (int componentIndex = 0; componentIndex < components.length; componentIndex++){
                    queryResult += "(";
                    var items = components[componentIndex].split(",\s*");
                    for (int itemIndex = 0; itemIndex < items.length; itemIndex++) {
                        String item = "'" + items[itemIndex].trim() + "'";
                        item = item.replaceAll("\\s*\\.\\s*", "");
                        queryResult += "game_on_platform( X," + item + ")";
                        if(itemIndex != items.length-1){
                            queryResult += "; ";
                        }
                    }
                    queryResult += ") ";
                    if(componentIndex != components.length-1){
                        queryResult += ", ";
                    }
                }
                queryResult += ") ";
            }
            if(queries[queryIndex].contains("графикой")){
                queries[queryIndex] = queries[queryIndex].substring(queries[queryIndex].indexOf(":")+1).trim();
                var components = queries[queryIndex].split("\s*и\s*");
                queryResult += "(";
                for (int componentIndex = 0; componentIndex < components.length; componentIndex++){
                    queryResult += "(";
                    var items = components[componentIndex].split(",\s*");
                    for (int itemIndex = 0; itemIndex < items.length; itemIndex++) {
                        String item = "'" + items[itemIndex].trim() + "'";
                        item = item.replaceAll("\\s*\\.\\s*", "");
                        queryResult += "game_graphics( X, " + item + ")";
                        if(itemIndex != items.length-1){
                            queryResult += "; ";
                        }
                    }
                    queryResult += ") ";
                    if(componentIndex != components.length-1){
                        queryResult += ", ";
                    }
                }
                queryResult += ") ";
            }
            if(queryIndex != queries.length-1){
                queryResult += ", ";
            }
        }
//        System.out.println(queryResult);
        return queryResult;
    }

}
