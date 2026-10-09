//
//  QueryableApp.swift
//  Queryable
//
//  Created by Mazzystar on 2023/07/09.
//

import SwiftUI

@main
struct QueryableApp: App {
    var body: some Scene {
        WindowGroup {
            if ProcessInfo.processInfo.environment["QUERYABLE_BENCHMARK"] == "1" {
                // The opt-in XCTest runner owns model loading and its synthetic
                // index. Avoid Photos prompts and a competing app index here.
                Color.clear
            } else {
                ContentView()
            }
        }
    }
}
